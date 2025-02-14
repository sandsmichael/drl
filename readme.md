Environment Setup (PortfolioEnv class):

Loads financial data from a CSV file
Sets up a sliding window of size 12 for observations
Separates feature columns (inputs) from forward return columns (portfolio holdings)
Implements reset/step mechanism for RL training
Neural Network Architecture:

Actor Network:
Input → 64 units → 64 units → Output (portfolio weights)
Uses ReLU activations and softmax at final layer
Outputs portfolio allocation weights that sum to 1
Critic Network:
Inputs both state and action
Concatenates them and processes through 64-unit layers
Outputs a single Q-value estimate
SAC Agent Implementation:

Uses both Actor and Target Critic networks
Maintains an experience replay buffer (max 100,000 experiences)
Batch size of 64 for training updates
Learning rate of 0.0003 and discount factor (gamma) of 0.99
Training Loop:

Runs for 500 episodes
For each episode:
Randomly selects a starting point in the data
Takes actions using the current policy
Stores experiences (state, action, reward, next_state, done)
Updates networks using SAC algorithm
Tracks rewards and portfolio allocations
Reward Calculation:

Based on Sharpe ratio: portfolio_return / portfolio_std
Uses forward 1-year returns for two assets (IWP and IWO)
Network Updates:

Uses two separate optimizers for Actor and Critic
Implements soft updates for target network
Handles missing next states in the experience replay
Evaluation:

Separate evaluation function to test the trained agent
Runs a single episode without training
Reports final Sharpe ratio
Portfolio Constraints:

Enforces non-negative weights (using clamp)
Ensures weights sum to 1 (using normalization)
Considers two assets for allocation (IWP and IWO)