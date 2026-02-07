# coding: utf-8
"""
Simple GraphSAGE Policy for VLSI Routing
Clean and minimal - just what you need!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import SAGEConv, global_mean_pool
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    print("Install: pip install torch torch-geometric")


class GraphSAGEPolicy(nn.Module):
    """
    Simple GraphSAGE for routing
    Clean, fast, effective!
    """
    
    def __init__(self, 
                 input_dim: int = 8,      # Node features
                 hidden_dim: int = 64,     # Hidden size
                 num_actions: int = 6):    # L, R, U, D, UP, DOWN
        super().__init__()
        
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("Need torch-geometric!")
        
        # GraphSAGE layers
        self.sage1 = SAGEConv(input_dim, hidden_dim)
        self.sage2 = SAGEConv(hidden_dim, hidden_dim)
        self.sage3 = SAGEConv(hidden_dim, hidden_dim)
        
        # Policy head (actor)
        self.policy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )
        
        # Value head (critic)
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, data):
        """
        Forward pass
        data: PyG Data object
        """
        x, edge_index = data.x, data.edge_index
        
        # GraphSAGE aggregation
        x = F.relu(self.sage1(x, edge_index))
        x = F.relu(self.sage2(x, edge_index))
        x = self.sage3(x, edge_index)
        
        # Global pooling
        graph_embedding = x.mean(dim=0, keepdim=True)
        
        # Policy and value
        action_logits = self.policy(graph_embedding)
        state_value = self.value(graph_embedding)
        
        return action_logits, state_value


class SimpleCNNPolicy(nn.Module):
    """
    Simple CNN policy (no GNN needed)
    Super fast baseline!
    """
    
    def __init__(self, 
                 input_channels: int = 5,
                 hidden_dim: int = 64,
                 num_actions: int = 6):
        super().__init__()
        
        # 3D CNN for grid
        self.conv1 = nn.Conv3d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool3d((1, 2, 2))
        
        # FC layers
        self.fc = nn.Sequential(
            nn.Linear(64 * 1 * 2 * 2, hidden_dim),
            nn.ReLU(),
        )
        
        # Policy head
        self.policy = nn.Linear(hidden_dim, num_actions)
        
        # Value head
        self.value = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        """
        x: [batch, layers, y, x, channels]
        """
        # Reshape to [batch, channels, layers, y, x]
        x = x.permute(0, 4, 1, 2, 3)
        
        # CNN
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        
        # FC
        x = self.fc(x)
        
        # Outputs
        action_logits = self.policy(x)
        state_value = self.value(x)
        
        return action_logits, state_value


class BasicRLAgent:
    """
    Simple RL agent for testing
    No complex stuff - just clean policy!
    """
    
    def __init__(self, policy, epsilon=0.1):
        self.policy = policy
        self.policy.eval()
        self.epsilon = epsilon  # Exploration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy.to(self.device)
    
    def select_action(self, observation, valid_actions):
        """
        Select action using policy
        
        Args:
            observation: State obs (numpy array or PyG Data)
            valid_actions: List of valid action strings
        
        Returns:
            action index
        """
        import random
        import numpy as np
        
        # Map actions to indices
        action_map = {
            'L': 0, 'R': 1, 'U': 2, 'D': 3,
            'UP_LAYER': 4, 'DOWN_LAYER': 5
        }
        
        valid_indices = [action_map[a] for a in valid_actions if a in action_map]
        if not valid_indices:
            return 0
        
        # Epsilon-greedy
        if random.random() < self.epsilon:
            return random.choice(valid_indices)
        
        # Policy
        with torch.no_grad():
            if isinstance(observation, np.ndarray):
                # CNN policy
                obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
                action_logits, _ = self.policy(obs_tensor)
            else:
                # GraphSAGE policy
                observation = observation.to(self.device)
                action_logits, _ = self.policy(observation)
            
            # Mask invalid actions
            action_probs = torch.softmax(action_logits[0], dim=-1).cpu().numpy()
            masked_probs = np.full(len(action_probs), -np.inf)
            for idx in valid_indices:
                masked_probs[idx] = action_probs[idx]
            
            return int(np.argmax(masked_probs))
    
    def save(self, path):
        """Save model"""
        torch.save(self.policy.state_dict(), path)
        print(f"Saved to {path}")
    
    def load(self, path):
        """Load model"""
        self.policy.load_state_dict(torch.load(path))
        print(f"Loaded from {path}")


def create_simple_agent(use_gnn=False, **kwargs):
    """
    Factory to create agent
    
    Args:
        use_gnn: True for GraphSAGE, False for CNN
    
    Returns:
        agent: BasicRLAgent
    """
    if use_gnn:
        print("Creating GraphSAGE agent...")
        policy = GraphSAGEPolicy(**kwargs)
    else:
        print("Creating CNN agent...")
        policy = SimpleCNNPolicy(**kwargs)
    
    agent = BasicRLAgent(policy)
    return agent


if __name__ == '__main__':
    # Quick test
    print("Testing GraphSAGE policy...")
    
    if TORCH_GEOMETRIC_AVAILABLE:
        policy = GraphSAGEPolicy(input_dim=8, hidden_dim=64, num_actions=6)
        print(f"GraphSAGE params: {sum(p.numel() for p in policy.parameters())}")
    
    policy = SimpleCNNPolicy(input_channels=5, hidden_dim=64, num_actions=6)
    print(f"CNN params: {sum(p.numel() for p in policy.parameters())}")
    
    # Test forward pass
    dummy_input = torch.randn(1, 2, 10, 10, 5)
    action_logits, value = policy(dummy_input)
    print(f"Output shapes: actions={action_logits.shape}, value={value.shape}")
    print("✓ All good!")
