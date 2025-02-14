import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt
import seaborn as sns

PORTFOLIO_HOLDINGS = [
    'IVV', 'AGG', 'TLT',
    'IWM', 'IWN', 'IWO',
    'IWB', 'IWF', 'IWD',
    'IWR', 'IWP', 'IWS', 
]

class PortfolioDataset:
    """
    Handles data preparation and loading for LSTM-based portfolio optimization.
    """
    def __init__(self, data_path="data.csv", window_size=12, portfolio_columns=None):
        self.data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        self.data.sort_index(inplace=True)
        
        self.portfolio_columns = portfolio_columns
        self.forward_columns = [f"{col}_fwd_1y" for col in portfolio_columns]
        self.feature_columns = [col for col in self.data.columns if "_fwd_" not in col]
        
        self.window_size = window_size
        self.state_size = len(self.feature_columns)
        self.num_assets = len(self.portfolio_columns)
        
        print("\nDataset Configuration:")
        print(f"Number of features: {self.state_size}")
        print(f"Window size: {window_size}")
        print(f"Number of assets: {self.num_assets}")
        
    def prepare_sequence(self, start_idx):
        """Prepare a single sequence of data for LSTM."""
        features = self.data.iloc[start_idx:start_idx + self.window_size][self.feature_columns].values
        returns = self.data.iloc[start_idx + self.window_size - 1][self.forward_columns].values
        return torch.FloatTensor(features), torch.FloatTensor(returns)
    
    def get_prediction_data(self):
        """Get the most recent window of data for prediction."""
        return self.prepare_sequence(len(self.data) - self.window_size)

class HybridPortfolioModel(nn.Module):
    """
    Hybrid LSTM model for portfolio optimization.
    Combines LSTM for market understanding with allocation head for portfolio weights.
    """
    def __init__(self, input_dim, hidden_dim, num_assets, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM for temporal pattern recognition
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        # Portfolio allocation head
        self.allocation = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_assets),
            nn.Softmax(dim=-1),
            # Add minimum allocation constraint
            MinMaxScaler(min_val=0.05, max_val=0.40)  # Min 5%, Max 40% per asset
        )
    
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # Use last LSTM output for allocation
        last_hidden = lstm_out[:, -1, :]
        # Generate portfolio weights
        weights = self.allocation(last_hidden)
        return weights

class MinMaxScaler(nn.Module):
    """Ensures allocations stay within specified bounds."""
    def __init__(self, min_val=0.05, max_val=0.40):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
    
    def forward(self, x):
        # Clip values
        x = torch.clamp(x, self.min_val, self.max_val)
        # Renormalize to sum to 1
        return x / x.sum(dim=-1, keepdim=True)


class PortfolioOptimizer:
    """
    Handles training and optimization of the hybrid portfolio model.
    """
    def __init__(self, dataset, hidden_dim=64, lr=0.001):
        self.dataset = dataset
        self.model = HybridPortfolioModel(
            input_dim=dataset.state_size,
            hidden_dim=hidden_dim,
            num_assets=dataset.num_assets
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
    def calculate_sharpe(self, weights, returns):
        """Calculate Sharpe ratio with diversification penalty."""
        portfolio_return = (weights * returns).sum()
        portfolio_std = torch.std(returns)
        
        # Add diversification penalty
        concentration_penalty = torch.sum(weights * weights)  # Penalize concentration
        entropy_penalty = -torch.sum(weights * torch.log(weights + 1e-6))  # Encourage diversity
        
        sharpe = portfolio_return / (portfolio_std + 1e-6)
        # Combine Sharpe ratio with diversification incentives
        adjusted_sharpe = sharpe + 0.1 * entropy_penalty - 0.1 * concentration_penalty
        
        return adjusted_sharpe
    
    def train_step(self, features, returns):
        """Execute one training step."""
        self.optimizer.zero_grad()
        
        # Get model prediction
        weights = self.model(features.unsqueeze(0))
        
        # Calculate Sharpe ratio (negative for minimization)
        loss = -self.calculate_sharpe(weights.squeeze(), returns)
        
        # Backpropagate and update
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), weights.detach().squeeze()

def train_model(num_episodes=25):
    """Train the hybrid portfolio model."""
    dataset = PortfolioDataset(data_path="data.csv", portfolio_columns=PORTFOLIO_HOLDINGS)
    optimizer = PortfolioOptimizer(dataset, hidden_dim=64, lr=0.0005)
    
    print("\nStarting Training...")
    rewards_history = []
    episode_losses = []  # Track losses for each episode
    
    for episode in range(num_episodes):
        total_reward = 0
        actions_taken = []
        losses = []  # Track losses within episode
        
        # Training loop
        for start_idx in range(len(dataset.data) - dataset.window_size - 252, 
                             len(dataset.data) - dataset.window_size):
            features, returns = dataset.prepare_sequence(start_idx)
            loss, weights = optimizer.train_step(features, returns)
            
            losses.append(loss)  # Store loss
            sharpe = -loss  # Convert loss back to Sharpe ratio
            total_reward += sharpe
            actions_taken.append(weights.numpy())
        
        # Track progress
        avg_loss = np.mean(losses)
        episode_losses.append(avg_loss)
        rewards_history.append(total_reward)
        avg_allocation = np.mean(actions_taken, axis=0)
        
        print(f"\nEpisode {episode + 1}")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Total Reward: {total_reward:.4f}")
        print(f"Average Portfolio Allocation:")
        for asset, weight in zip(PORTFOLIO_HOLDINGS, avg_allocation):
            print(f"{asset:4s}: {weight:7.2%}")
        print("-" * 40)
    
    # Plot training losses
    plt.figure(figsize=(10, 5))
    plt.plot(episode_losses, label='Average Loss per Episode')
    plt.title('Training Loss Over Time')
    plt.xlabel('Episode')
    plt.ylabel('Loss (Negative Sharpe Ratio)')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_loss.png')
    plt.close()
    
# In train_model function, update the save operation:
    # Save trained model with loss history
    torch.save({
        'model_state_dict': optimizer.model.state_dict(),
        'loss_history': episode_losses,
        'rewards_history': rewards_history.copy()  # Use .copy() for numpy arrays
    }, 'trained_portfolio_lstm.pth', _use_new_zipfile_serialization=True)
    
    return optimizer, episode_losses

def get_current_allocation(model_path='trained_portfolio_lstm.pth'):
    """Get current portfolio allocation prediction."""
    dataset = PortfolioDataset(data_path="data.csv", portfolio_columns=PORTFOLIO_HOLDINGS)
    model = HybridPortfolioModel(dataset.state_size, 64, dataset.num_assets)
    
    # Add numpy scalar to safe globals
    import numpy._core.multiarray
    torch.serialization.add_safe_globals([
        numpy._core.multiarray.scalar
    ])
    
    # Load trained model
    try:
        checkpoint = torch.load(model_path, weights_only=False)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Ensuring model exists...")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found. Run training first.")
        return None
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Get prediction
    features, _ = dataset.get_prediction_data()
    with torch.no_grad():
        weights = model(features.unsqueeze(0))
    
    # Format results
    portfolio = {
        asset: float(weight)
        for asset, weight in zip(PORTFOLIO_HOLDINGS, weights[0])
    }
    
    print(f"\nAllocation as of: {dataset.data.index[-1]}")
    return portfolio



def analyze_lstm_model(model_path='trained_portfolio_lstm.pth'):
    """Analyze the trained LSTM model."""
    dataset = PortfolioDataset(data_path="data.csv", portfolio_columns=PORTFOLIO_HOLDINGS)
    model = HybridPortfolioModel(dataset.state_size, 64, dataset.num_assets)
    
    checkpoint = torch.load(model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("\nLSTM Model Analysis:")
    print("-" * 40)
    
    # Analyze LSTM weights
    lstm_weights = {name: param for name, param in model.named_parameters() if 'lstm' in name}
    for name, weights in lstm_weights.items():
        print(f"\n{name}:")
        print(f"Shape: {weights.shape}")
        print(f"Mean: {weights.mean().item():.4f}")
        print(f"Std: {weights.std().item():.4f}")
    
    # Analyze allocation head
    allocation_weights = {name: param for name, param in model.named_parameters() if 'allocation' in name}
    for name, weights in allocation_weights.items():
        print(f"\n{name}:")
        print(f"Shape: {weights.shape}")
        print(f"Mean: {weights.mean().item():.4f}")
        print(f"Std: {weights.std().item():.4f}")

def evaluate_model_performance(model_path='trained_portfolio_lstm.pth'):
    """
    Evaluate the trained LSTM model's performance using various metrics:
    - Rolling Sharpe Ratio
    - Portfolio Returns
    - Allocation Stability
    - Diversification Metrics
    - Out-of-sample Performance
    """
    # Load dataset and model
    dataset = PortfolioDataset(data_path="data.csv", portfolio_columns=PORTFOLIO_HOLDINGS)
    model = HybridPortfolioModel(dataset.state_size, 64, dataset.num_assets)
    checkpoint = torch.load(model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Storage for metrics
    allocations = []
    returns = []
    sharpe_ratios = []
    rolling_window = 21  # About 1 month of trading days

    print("\nModel Performance Evaluation:")
    print("-" * 60)

    # Evaluate on each timestep
    for start_idx in range(len(dataset.data) - dataset.window_size - 252, 
                          len(dataset.data) - dataset.window_size):
        features, forward_returns = dataset.prepare_sequence(start_idx)
        
        # Get model prediction
        with torch.no_grad():
            weights = model(features.unsqueeze(0))
            weights = weights.squeeze().numpy()
        
        # Store results
        allocations.append(weights)
        portfolio_return = np.sum(weights * forward_returns.numpy())
        returns.append(portfolio_return)

    # Convert to numpy arrays
    allocations = np.array(allocations)
    returns = np.array(returns)

    # Calculate metrics
    # 1. Portfolio Statistics
    total_return = np.prod(1 + returns) - 1
    annual_return = (1 + total_return) ** (252 / len(returns)) - 1
    volatility = np.std(returns) * np.sqrt(252)
    sharpe = annual_return / volatility
    
    # 2. Diversification Metrics
    avg_allocation = np.mean(allocations, axis=0)
    allocation_std = np.std(allocations, axis=0)
    min_allocation = np.min(allocations, axis=0)
    max_allocation = np.max(allocations, axis=0)
    
    # 3. Turnover Analysis
    turnover = np.mean(np.abs(allocations[1:] - allocations[:-1]).sum(axis=1))

    # Print Results
    print("\n1. Portfolio Performance:")
    print(f"Total Return: {total_return:.2%}")
    print(f"Annualized Return: {annual_return:.2%}")
    print(f"Annualized Volatility: {volatility:.2%}")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    
    print("\n2. Allocation Statistics:")
    print("Average Allocations:")
    for asset, alloc, std in zip(PORTFOLIO_HOLDINGS, avg_allocation, allocation_std):
        print(f"{asset:4s}: {alloc:7.2%} (±{std:.2%})")
    
    print("\n3. Portfolio Characteristics:")
    print(f"Average Turnover: {turnover:.2%}")
    effective_n = 1 / np.sum(avg_allocation ** 2)
    print(f"Effective N (Diversification): {effective_n:.1f}")

    # Visualize allocation stability
    plt.figure(figsize=(15, 8))
    plt.stackplot(range(len(allocations)), allocations.T, 
                 labels=PORTFOLIO_HOLDINGS)
    plt.title('Portfolio Allocation Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Allocation')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('allocation_history.png')
    plt.close()

    # Visualize returns distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(returns, kde=True)
    plt.title('Distribution of Portfolio Returns')
    plt.xlabel('Return')
    plt.ylabel('Frequency')
    plt.savefig('returns_distribution.png')
    plt.close()

    return {
        'sharpe_ratio': sharpe,
        'annual_return': annual_return,
        'volatility': volatility,
        'turnover': turnover,
        'effective_n': effective_n
    }

import os
def main():
    ### Training
    if not os.path.exists('trained_portfolio_lstm.pth'):
        print("No trained model found. Starting training...")
        ### Training
        train_model()    
    
    ### Inference
    allocation = get_current_allocation()
    
    print("\nFinal Portfolio Allocation:")
    print("-" * 40)
    for asset, weight in sorted(allocation.items()):
        print(f"{asset:4s}: {weight:7.2%}")
    print("-" * 40)
    
    ### Analysis
    # analyze_lstm_model()

    evaluate_model_performance()

main()









# Add these imports at the top
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.preprocessing import StandardScaler

def analyze_lstm_decisions(model_path='trained_portfolio_lstm.pth', max_depth=3):
    """
    Analyze LSTM decisions by fitting an interpretable decision tree
    to the LSTM's behavior.
    """
    # Load dataset and model
    dataset = PortfolioDataset(data_path="data.csv", portfolio_columns=PORTFOLIO_HOLDINGS)
    model = HybridPortfolioModel(dataset.state_size, 64, dataset.num_assets)
    checkpoint = torch.load(model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Collect LSTM predictions and corresponding features
    X_train = []
    y_train = []
    
    for start_idx in range(len(dataset.data) - dataset.window_size - 252, 
                          len(dataset.data) - dataset.window_size):
        features, _ = dataset.prepare_sequence(start_idx)
        
        # Get LSTM prediction
        with torch.no_grad():
            weights = model(features.unsqueeze(0))
            weights = weights.squeeze().numpy()
        
        # Store last timepoint features and predictions
        X_train.append(features[-1].numpy())  # Use last timestep features
        y_train.append(weights)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Create decision trees for each asset
    print("\nDecision Tree Analysis:")
    print("-" * 60)
    
    for asset_idx, asset in enumerate(PORTFOLIO_HOLDINGS):
        # Fit decision tree for this asset's allocation
        tree = DecisionTreeRegressor(max_depth=max_depth)
        tree.fit(X_train_scaled, y_train[:, asset_idx])
        
        # Get feature importance
        importances = tree.feature_importances_
        feature_imp = list(zip(dataset.feature_columns, importances))
        feature_imp.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n{asset} Allocation Rules:")
        print("-" * 40)
        
        # Print top 5 important features
        print("Top 5 Important Features:")
        for feature, importance in feature_imp[:5]:
            print(f"{feature}: {importance:.4f}")
        
        # Print decision tree rules
        print("\nDecision Rules:")
        tree_rules = export_text(tree, 
                               feature_names=dataset.feature_columns,
                               show_weights=True)
        print(tree_rules)
        
        # Print example prediction
        current_features = X_train_scaled[-1].reshape(1, -1)
        tree_pred = tree.predict(current_features)[0]
        lstm_pred = y_train[-1, asset_idx]
        print(f"\nLatest Prediction:")
        print(f"Tree: {tree_pred:.2%}")
        print(f"LSTM: {lstm_pred:.2%}")

# analyze_lstm_decisions(max_depth=3)
