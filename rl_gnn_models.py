# coding: utf-8
"""
Graph Neural Network Models for VLSI Routing

Implements GNN architectures for processing graph-based routing states.
Can be used with Deep RL algorithms (DQN, PPO, A3C, etc.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Tuple

try:
    from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool, global_max_pool
    from torch_geometric.data import Data, Batch
    PYTORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    PYTORCH_GEOMETRIC_AVAILABLE = False
    print("Warning: PyTorch Geometric not available. GNN models will not work.")


class GCNRoutingPolicy(nn.Module):
    """
    Graph Convolutional Network for routing policy
    Uses GCN layers to process routing graph state
    """
    
    def __init__(self,
                 node_feature_dim: int = 11,  # From GCellNode.to_feature_vector()
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 num_actions: int = 6,
                 dropout: float = 0.1):
        """
        Initialize GCN routing policy
        
        Args:
            node_feature_dim: Dimension of node features
            hidden_dim: Hidden layer dimension
            num_layers: Number of GCN layers
            num_actions: Number of possible actions
            dropout: Dropout rate
        """
        super(GCNRoutingPolicy, self).__init__()
        
        if not PYTORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("PyTorch Geometric required for GNN models")
        
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_actions = num_actions
        
        # GCN layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(node_feature_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        self.dropout = dropout
        
        # Policy head (actor)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_actions),
        )
        
        # Value head (critic) for actor-critic methods
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, data: Data, current_node_idx: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            data: PyTorch Geometric Data object
            current_node_idx: Index of current node (for node-specific embeddings)
            
        Returns:
            action_logits: Logits for action distribution
            value: State value estimate
        """
        x, edge_index = data.x, data.edge_index
        
        # Apply GCN layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling for graph-level representation
        graph_embedding = global_mean_pool(x, data.batch) if hasattr(data, 'batch') else x.mean(dim=0, keepdim=True)
        
        # If current node specified, use its embedding
        if current_node_idx is not None:
            node_embedding = x[current_node_idx]
            # Combine graph and node embeddings
            combined = graph_embedding + node_embedding
        else:
            combined = graph_embedding
        
        # Policy and value outputs
        action_logits = self.policy_head(combined)
        value = self.value_head(combined)
        
        return action_logits, value


class GATRoutingPolicy(nn.Module):
    """
    Graph Attention Network for routing policy
    Uses attention mechanism to focus on important nodes
    """
    
    def __init__(self,
                 node_feature_dim: int = 11,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 num_heads: int = 4,
                 num_actions: int = 6,
                 dropout: float = 0.1):
        """
        Initialize GAT routing policy
        
        Args:
            node_feature_dim: Dimension of node features
            hidden_dim: Hidden layer dimension
            num_layers: Number of GAT layers
            num_heads: Number of attention heads
            num_actions: Number of possible actions
            dropout: Dropout rate
        """
        super(GATRoutingPolicy, self).__init__()
        
        if not PYTORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("PyTorch Geometric required for GNN models")
        
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_actions = num_actions
        
        # GAT layers
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(node_feature_dim, hidden_dim, heads=num_heads, dropout=dropout))
        
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout))
        
        # Last layer with single head
        self.convs.append(GATConv(hidden_dim * num_heads, hidden_dim, heads=1, dropout=dropout))
        
        self.dropout = dropout
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_actions),
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, data: Data, current_node_idx: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through GAT"""
        x, edge_index = data.x, data.edge_index
        
        # Apply GAT layers
        for i, conv in enumerate(self.convs):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x)
        
        # Global pooling
        graph_embedding = global_mean_pool(x, data.batch) if hasattr(data, 'batch') else x.mean(dim=0, keepdim=True)
        
        # Node-specific embedding if specified
        if current_node_idx is not None:
            node_embedding = x[current_node_idx]
            combined = graph_embedding + node_embedding
        else:
            combined = graph_embedding
        
        # Outputs
        action_logits = self.policy_head(combined)
        value = self.value_head(combined)
        
        return action_logits, value


class CNNRoutingPolicy(nn.Module):
    """
    Convolutional Neural Network for routing policy
    Processes grid-based state representation (not graph-based)
    Useful when graph processing is not available
    """
    
    def __init__(self,
                 input_channels: int = 7,  # From to_feature_matrix()
                 num_layers: int = 5,
                 num_actions: int = 6,
                 hidden_dim: int = 128):
        """
        Initialize CNN routing policy
        
        Args:
            input_channels: Number of feature channels
            num_layers: Number of layers
            num_actions: Number of possible actions
            hidden_dim: Hidden dimension
        """
        super(CNNRoutingPolicy, self).__init__()
        
        self.input_channels = input_channels
        self.num_actions = num_actions
        
        # 3D CNN for processing layer-wise features
        # Input: [batch, channels, layers, y, x]
        self.conv_layers = nn.ModuleList()
        
        # First conv layer
        self.conv_layers.append(nn.Conv3d(input_channels, 64, kernel_size=3, padding=1))
        self.conv_layers.append(nn.Conv3d(64, 64, kernel_size=3, padding=1))
        self.conv_layers.append(nn.Conv3d(64, 128, kernel_size=3, padding=1))
        self.conv_layers.append(nn.Conv3d(128, 128, kernel_size=3, padding=1))
        
        # Adaptive pooling to fixed size
        self.adaptive_pool = nn.AdaptiveAvgPool3d((2, 4, 4))
        
        # Flatten dimension
        flatten_dim = 128 * 2 * 4 * 4
        
        # Fully connected layers
        self.fc_shared = nn.Sequential(
            nn.Linear(flatten_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor [batch, layers, y, x, channels]
            
        Returns:
            action_logits: Logits for actions
            value: State value
        """
        # Reshape to [batch, channels, layers, y, x]
        x = x.permute(0, 4, 1, 2, 3)
        
        # Apply conv layers
        for conv in self.conv_layers:
            x = F.relu(conv(x))
        
        # Pool and flatten
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        
        # Shared FC
        x = self.fc_shared(x)
        
        # Separate heads
        action_logits = self.policy_head(x)
        value = self.value_head(x)
        
        return action_logits, value


class DuelingDQN(nn.Module):
    """
    Dueling DQN architecture for Q-learning
    Separates value and advantage streams
    """
    
    def __init__(self,
                 node_feature_dim: int = 11,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 num_actions: int = 6,
                 use_gnn: bool = True):
        """
        Initialize Dueling DQN
        
        Args:
            node_feature_dim: Node feature dimension
            hidden_dim: Hidden dimension
            num_layers: Number of layers
            num_actions: Number of actions
            use_gnn: Whether to use GNN or CNN backbone
        """
        super(DuelingDQN, self).__init__()
        
        self.use_gnn = use_gnn
        self.num_actions = num_actions
        
        if use_gnn:
            if not PYTORCH_GEOMETRIC_AVAILABLE:
                raise ImportError("PyTorch Geometric required for GNN")
            
            # GNN backbone
            self.convs = nn.ModuleList()
            self.convs.append(GCNConv(node_feature_dim, hidden_dim))
            for _ in range(num_layers - 1):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
        else:
            # CNN backbone
            self.cnn = CNNRoutingPolicy(num_actions=num_actions, hidden_dim=hidden_dim)
            hidden_dim = 128  # From CNN output
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )
    
    def forward(self, x, edge_index=None):
        """Forward pass"""
        if self.use_gnn:
            # GNN processing
            for conv in self.convs:
                x = F.relu(conv(x, edge_index))
            x = global_mean_pool(x, torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        else:
            # CNN processing
            x, _ = self.cnn(x)
        
        # Dueling streams
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        # Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        
        return q_values


class MultiHeadAttentionRouting(nn.Module):
    """
    Advanced model using multi-head attention
    for routing-specific attention mechanisms
    """
    
    def __init__(self,
                 node_feature_dim: int = 11,
                 hidden_dim: int = 128,
                 num_heads: int = 8,
                 num_actions: int = 6):
        """Initialize attention-based routing model"""
        super(MultiHeadAttentionRouting, self).__init__()
        
        self.node_encoder = nn.Linear(node_feature_dim, hidden_dim)
        
        # Multi-head attention for node relationships
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Position-aware encoding
        self.position_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim // 2),  # 3 for (layer, y, x)
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # Policy network
        self.policy = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_actions)
        )
        
        # Value network
        self.value = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, node_features, positions, current_idx):
        """
        Forward pass with attention
        
        Args:
            node_features: Node feature tensor [num_nodes, feature_dim]
            positions: Node positions [num_nodes, 3] (layer, y, x)
            current_idx: Index of current node
            
        Returns:
            action_logits, value
        """
        # Encode node features
        node_embed = self.node_encoder(node_features)
        
        # Encode positions
        pos_embed = self.position_encoder(positions.float())
        
        # Combine features and positions
        combined = node_embed + pos_embed
        
        # Self-attention over all nodes
        attended, _ = self.attention(
            combined.unsqueeze(0),
            combined.unsqueeze(0),
            combined.unsqueeze(0)
        )
        attended = attended.squeeze(0)
        
        # Use current node embedding
        current_embed = attended[current_idx]
        global_context = attended.mean(dim=0)
        
        # Concatenate for final prediction
        final_embed = torch.cat([current_embed, global_context], dim=-1)
        
        # Outputs
        action_logits = self.policy(final_embed)
        value = self.value(final_embed)
        
        return action_logits.unsqueeze(0), value.unsqueeze(0)


def create_model(model_type: str = 'gcn', **kwargs):
    """
    Factory function to create routing models
    
    Args:
        model_type: Type of model ('gcn', 'gat', 'cnn', 'dueling_dqn', 'attention')
        **kwargs: Model-specific arguments
        
    Returns:
        model: PyTorch model
    """
    models = {
        'gcn': GCNRoutingPolicy,
        'gat': GATRoutingPolicy,
        'cnn': CNNRoutingPolicy,
        'dueling_dqn': DuelingDQN,
        'attention': MultiHeadAttentionRouting,
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(models.keys())}")
    
    return models[model_type](**kwargs)
