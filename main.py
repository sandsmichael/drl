import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque


PORTFOLIO_HOLDINGS = [
    'IVV', 'AGG', 'TLT',
    'IWM', 'IWN', 'IWO',
    'IWB', 'IWF', 'IWD',
    'IWR', 'IWP', 'IWS', 
 ]

# ---------------- 1. Load Data ----------------
class PortfolioEnv:
    """
    A custom environment for portfolio optimization using reinforcement learning.
    
    This environment simulates portfolio allocation across multiple assets using historical data.
    It provides a gym-like interface with step() and reset() methods, uses a sliding window
    of market data as state, and calculates rewards based on forward returns.
    
    Attributes:
        data (pd.DataFrame): Historical market data including features and forward returns
        window_size (int): Number of time steps to include in each state observation
        portfolio_columns (list): Names of assets available for trading
        forward_columns (list): Names of corresponding forward return columns
        feature_columns (list): Names of all input features (excluding forward returns)
        state_size (int): Total dimension of state space (features × window_size)
        num_assets (int): Number of assets available for trading
    """    
    def __init__(self, data_path="data.csv", window_size=12, portfolio_columns=None):
        self.data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        self.data.sort_index(inplace=True)
        
        # Setup portfolio columns (the assets we can trade)
        self.portfolio_columns = portfolio_columns
        # Create mapping to forward return columns
        self.forward_columns = [f"{col}_fwd_1y" for col in portfolio_columns]
        
        # Verify all columns exist in data
        missing_cols = [col for col in self.forward_columns if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Forward return columns not found in data: {missing_cols}")
        
        # Setup feature columns (all columns except forward returns)
        self.feature_columns = [col for col in self.data.columns if "_fwd_" not in col]
        
        # Rest of initialization remains the same
        self.window_size = window_size
        self.start_idx = window_size
        self.end_idx = len(self.data) - 1
        self.possible_starts = range(self.start_idx, self.end_idx - 252)
        self.state_size = len(self.feature_columns) * window_size
        self.num_assets = len(self.portfolio_columns)
        
        # Print setup information
        print("\nEnvironment Configuration:")
        print(f"Number of features: {len(self.feature_columns)}")
        print(f"State size (features × window): {self.state_size}")
        print(f"Number of tradeable assets: {self.num_assets}")
        print(f"Feature columns include: {', '.join(self.feature_columns[:5])}...")
        print(f"Trading portfolio: {', '.join(self.portfolio_columns)}")
        print(f"Forward return columns: {', '.join(self.forward_columns)}")

    def step(self, action):
        """
        Execute one time step within the environment.
        
        Takes a portfolio allocation action and moves to the next time step, calculating
        the reward based on forward returns. The reward is the Sharpe ratio of the portfolio,
        calculated as the portfolio return divided by its standard deviation.
        
        Args:
            action (numpy.array): Portfolio weights summing to 1, shape (num_assets,)
        
        Returns:
            tuple: (next_state, reward, done) where:
                - next_state: New observation of market features
                - reward: Sharpe ratio of the portfolio allocation
                - done: Whether this episode has ended
        """        
        """Takes an action (allocation weights), returns next state, reward, and done."""
        action = torch.tensor(action, dtype=torch.float32)
        action = torch.clamp(action, 0, 1)
        action /= action.sum() + 1e-6

        # Use forward return columns for reward calculation
        forward_returns = self.data.iloc[self.current_step][self.forward_columns].values
        forward_returns = torch.tensor(forward_returns, dtype=torch.float32)
        portfolio_return = torch.dot(action, forward_returns)
        portfolio_std = torch.std(forward_returns)
        reward = portfolio_return / (portfolio_std + 1e-6)  # Sharpe ratio

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        next_state = self._get_observation() if not done else None

        return next_state, reward, done


    def _get_observation(self):
        """
        Construct the current state observation from historical data.
        
        Creates a state vector by taking a sliding window of historical features
        and flattening them into a single vector. The window includes the past
        window_size time steps of all feature columns.
        
        Returns:
            torch.Tensor: Flattened vector of historical features, shape (state_size,)
        """
        start = self.current_step - self.window_size
        obs = self.data.iloc[start:self.current_step][self.feature_columns].values.flatten()
        return torch.tensor(obs, dtype=torch.float32)

    def reset(self):
        """
        Reset the environment to start a new episode.
        
        Randomly selects a starting point in the data to enable learning from
        different market conditions. Initializes portfolio weights to equal allocation.
        
        Returns:
            torch.Tensor: Initial state observation
        """
        # Randomly select a starting point
        self.current_step = random.choice(list(self.possible_starts))
        self.weights = np.ones(self.num_assets) / self.num_assets
        return self._get_observation()




# ---------------- 2. SAC Model ----------------
class Actor(nn.Module):
    """
    Neural network that learns the portfolio allocation policy.
    
    Maps state observations to portfolio weights using a feedforward neural network.
    Uses softmax activation on the output layer to ensure valid portfolio weights
    that sum to 1.
    
    Attributes:
        state_dim (int): Dimension of input state
        action_dim (int): Number of assets to allocate between
        fc1, fc2, fc3 (nn.Linear): Fully connected layers
    """
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, state):
        """
        Forward pass through the network to generate portfolio weights.
        
        Args:
            state (torch.Tensor): Current market state observation
            
        Returns:
            torch.Tensor: Portfolio allocation weights that sum to 1
        """        
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)  # Ensures valid portfolio weights

class Critic(nn.Module):
    """
    Neural network that estimates state-action values.
    
    Takes both state and action as input and estimates their Q-value,
    which represents the expected future rewards for taking that action
    in that state.
    
    Attributes:
        state_dim (int): Dimension of input state
        action_dim (int): Dimension of action space
        fc1, fc2, fc3 (nn.Linear): Fully connected layers
    """
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, state, action):
        """
        Forward pass through the network to estimate Q-value.
        
        Args:
            state (torch.Tensor): Current market state observation
            action (torch.Tensor): Portfolio allocation action
            
        Returns:
            torch.Tensor: Estimated Q-value for the state-action pair
        """        
        x = torch.cat([state, action], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# ---------------- 3. SAC Training ----------------
class SACAgent:
    """
    Soft Actor-Critic agent for portfolio optimization.
    
    Implements the SAC algorithm with experience replay and target networks.
    Uses separate actor and critic networks to learn both a policy and
    value function, with entropy regularization for better exploration.
    
    Attributes:
        actor (Actor): Policy network for selecting actions
        critic (Critic): Q-value network for evaluating actions
        target_critic (Critic): Slowly-updating target network for stability
        memory (deque): Replay buffer storing past experiences
        exploration_noise (float): Standard deviation of Gaussian exploration noise
    """
    def __init__(self, state_dim, action_dim, lr=0.0003, gamma=0.99):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim, action_dim)
        self.target_critic = Critic(state_dim, action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.gamma = gamma
        self.memory = deque(maxlen=100000)
        self.batch_size = 64
        self.exploration_noise = 0.1  # Add exploration noise parameter

    def select_action(self, state, explore=True):
        """
        Select a portfolio allocation action given the current state.
        
        Uses the actor network to generate base action and optionally adds
        exploration noise. Ensures valid portfolio weights by clipping to [0,1]
        and normalizing to sum to 1.
        
        Args:
            state (torch.Tensor): Current market state observation
            explore (bool): Whether to add exploration noise
            
        Returns:
            numpy.array: Portfolio allocation weights
        """
        with torch.no_grad():
            action = self.actor(state).numpy()
            if explore:
                # Add Gaussian noise for exploration
                noise = np.random.normal(0, self.exploration_noise, size=action.shape)
                action = action + noise
                # Ensure valid portfolio weights after adding noise
                action = np.clip(action, 0, 1)
                action = action / (action.sum() + 1e-6)
            return action

    def store_experience(self, state, action, reward, next_state, done):
        """
        Store a transition in the replay buffer.
        
        Args:
            state: Current state observation
            action: Action taken
            reward: Reward received
            next_state: Next state observation
            done: Whether episode ended
        """
        self.memory.append((state, action, reward, next_state, done))

    def update(self):
        """
        Update the actor and critic networks using sampled experiences.
        
        Implements the core SAC learning algorithm:
        1. Samples a batch of experiences from replay buffer
        2. Computes target Q-values using target network
        3. Updates critic to minimize TD error
        4. Updates actor to maximize expected Q-value
        5. Soft updates target network
        """
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Filter out None values from next_states
        valid_next = [(i, ns) for i, ns in enumerate(next_states) if ns is not None]
        valid_indices = [i for i, _ in valid_next]
        valid_next_states = [ns for _, ns in valid_next]

        # Convert to tensors properly
        states = torch.stack([s.clone().detach() for s in states])
        actions = torch.stack([torch.tensor(a, dtype=torch.float32) for a in actions])
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.stack([s.clone().detach() for s in valid_next_states])
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        # Critic Loss
        with torch.no_grad():
            next_actions = self.actor(next_states)
            target_q_next = self.target_critic(next_states, next_actions)
            
            # Initialize target_q with rewards
            target_q = rewards.clone()
            # Update only the valid indices
            target_q[valid_indices] += self.gamma * target_q_next * (1 - dones[valid_indices])

        q_value = self.critic(states, actions)
        critic_loss = nn.MSELoss()(q_value, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor Loss
        new_actions = self.actor(states)
        actor_loss = -self.critic(states, new_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target critic
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(0.995 * target_param.data + 0.005 * param.data)

# ---------------- 4. Train & Evaluate ----------------
def train_agent():
    """
    Train the SAC agent on the portfolio optimization task.
    
    Creates environment and agent, then runs training loop for specified
    number of episodes. Implements exploration noise decay and tracks
    rewards and allocations over time. Prints periodic updates on
    training progress.
    
    Returns:
        SACAgent: Trained agent
    """    
    env = PortfolioEnv(data_path="data.csv", portfolio_columns=PORTFOLIO_HOLDINGS)
    agent = SACAgent(state_dim=env.state_size, action_dim=env.num_assets)
    
    num_episodes = 25
    rewards_history = []
    exploration_decay = 0.995  # Decay exploration over time
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        actions_taken = []

        # Decay exploration noise
        agent.exploration_noise *= exploration_decay
        agent.exploration_noise = max(0.01, agent.exploration_noise)  # Don't go below 0.01

        while not done:
            action = agent.select_action(state, explore=True)  # Enable exploration
            actions_taken.append(action)
            next_state, reward, done = env.step(action)

            agent.store_experience(state, action, reward, next_state, done)
            agent.update()

            state = next_state
            total_reward += reward
        
        rewards_history.append(total_reward)
        avg_reward = np.mean(rewards_history[-100:]) if len(rewards_history) >= 100 else np.mean(rewards_history)
        
        # if episode % 10 == 0:  # print out every 10 runs
        avg_allocation = np.mean(actions_taken, axis=0)
        print(f"Episode {episode + 1}")
        print(f"Total Reward: {total_reward:.4f}")
        print(f"100-episode Average Reward: {avg_reward:.4f}")
        print(f"Average Portfolio Allocation: {avg_allocation}")
        print("-" * 50)
    
    save_trained_agent(agent)


def save_trained_agent(agent, filepath='trained_portfolio_agent.pth'):
    """Save the trained agent's parameters"""
    torch.save({
        'actor_state_dict': agent.actor.state_dict(),
        'critic_state_dict': agent.critic.state_dict(),
        'target_critic_state_dict': agent.target_critic.state_dict(),
    }, filepath)


def evaluate_agent():
    """
    Evaluate a trained agent's performance.
    
    Creates a new environment instance and runs one full episode
    without exploration noise. Tracks and reports the achieved
    Sharpe ratio and average portfolio allocations.
    
    Returns:
        float: Total reward (Sharpe ratio) achieved
    """    
    env = PortfolioEnv(data_path="data.csv", portfolio_columns=PORTFOLIO_HOLDINGS)
    agent = SACAgent(state_dim=env.state_size, action_dim=env.num_assets)

    state = env.reset()
    done = False
    total_reward = 0
    allocations = []

    while not done:
        action = agent.select_action(state, explore=False)  # Disable exploration for evaluation
        allocations.append(action)
        state, reward, done = env.step(action)
        total_reward += reward

    avg_allocation = np.mean(allocations, axis=0)
    print(f"Total Reward (Sharpe Ratio): {total_reward:.4f}")
    print("\nAverage Portfolio Allocation:")
    for asset, alloc in zip(PORTFOLIO_HOLDINGS, avg_allocation):
        print(f"{asset}: {alloc:.2%}")

""" 
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ Train & Evaluate                                                                                                 │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
"""

train_agent()
evaluate_agent()






""" 
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ Predict                                                                                                          │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
"""

def prepare_current_data(raw_data_path, output_path):
    """
    Prepare current market data in the format expected by the model.
    """
    # Read and process current market data
    data = pd.read_csv(raw_data_path)
    
    # Ensure all required columns exist
    required_columns = [col for col in PORTFOLIO_HOLDINGS]
    missing = [col for col in required_columns if col not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Add placeholder forward return columns
    for col in PORTFOLIO_HOLDINGS:
        data[f"{col}_fwd_1y"] = 0  # placeholder, not used for prediction
    
    # Save processed data
    data.to_csv(output_path)
    return output_path



def get_current_allocation(new_data_path, model_path='trained_portfolio_agent.pth'):
    """
    Get portfolio allocation suggestion for current market conditions.
    
    Args:
        new_data_path: Path to CSV with recent market data
        model_path: Path to saved model parameters
    
    Returns:
        dict: Mapping of assets to suggested allocation weights
    """
    # Create environment with same parameters as training
    env = PortfolioEnv(
        data_path=new_data_path,
        portfolio_columns=PORTFOLIO_HOLDINGS
    )
    
    # Create agent with same architecture
    agent = SACAgent(state_dim=env.state_size, action_dim=env.num_assets)
    
    # Load trained parameters
    checkpoint = torch.load(model_path)
    agent.actor.load_state_dict(checkpoint['actor_state_dict'])
    agent.actor.eval()  # Set to evaluation mode
    
    # Get current state
    state = env._get_observation()
    
    # Get allocation without exploration
    with torch.no_grad():
        allocation = agent.select_action(state, explore=False)
    
    # Create dictionary of allocations
    portfolio = {
        asset: float(weight) 
        for asset, weight in zip(PORTFOLIO_HOLDINGS, allocation)
    }
    
    return portfolio