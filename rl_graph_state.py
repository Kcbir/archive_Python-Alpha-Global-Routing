# coding: utf-8
"""
Graph-based State Representation for VLSI Routing RL
This module represents the routing state as a graph/heterograph structure
suitable for GNN processing and RL decision making.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import torch

try:
    import rustworkx as rx
    RUSTWORKX_AVAILABLE = True
except ImportError:
    RUSTWORKX_AVAILABLE = False
    print("Warning: Rustworkx not available. Install with: pip install rustworkx")

try:
    import torch_geometric
    from torch_geometric.data import Data
    PYTORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    PYTORCH_GEOMETRIC_AVAILABLE = False
    print("Warning: PyTorch Geometric not available. Install with: pip install torch-geometric")


@dataclass
class GCellNode:
    """Represents a single GCell in the routing grid"""
    layer: int
    x: int
    y: int
    capacity: float  # Remaining capacity
    original_capacity: float  # Original capacity
    congestion: float  # Current congestion level (0-1)
    is_blocked: bool  # Whether blocked by macro
    is_terminal: bool  # Whether it's a terminal for current net
    is_source: bool  # Whether it's source for current net
    is_target: bool  # Whether it's target for current net
    occupancy: int  # Number of nets using this cell
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert node to feature vector for GNN"""
        return np.array([
            self.layer,
            self.x,
            self.y,
            self.capacity,
            self.original_capacity,
            self.congestion,
            float(self.is_blocked),
            float(self.is_terminal),
            float(self.is_source),
            float(self.is_target),
            self.occupancy,
        ], dtype=np.float32)


@dataclass
class EdgeInfo:
    """Represents an edge between two GCells"""
    node1: Tuple[int, int, int]  # (layer, y, x)
    node2: Tuple[int, int, int]  # (layer, y, x)
    edge_type: str  # 'horizontal', 'vertical', 'via'
    capacity: float
    congestion: float
    cost: float
    is_occupied: bool
    occupancy_count: int
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert edge to feature vector"""
        edge_type_encoding = {
            'horizontal': 0,
            'vertical': 1,
            'via': 2
        }
        return np.array([
            edge_type_encoding[self.edge_type],
            self.capacity,
            self.congestion,
            self.cost,
            float(self.is_occupied),
            self.occupancy_count,
        ], dtype=np.float32)


class RoutingGraphState:
    """
    Graph-based state representation for VLSI routing
    Supports both NetworkX graphs and PyTorch Geometric for GNN
    """
    
    def __init__(self, capacity_matrix: np.ndarray, layer_directions: List[int]):
        """
        Initialize the routing graph state
        
        Args:
            capacity_matrix: 3D array [layers, y, x] with capacity values
            layer_directions: List indicating routing direction per layer (0=horizontal, 1=vertical)
        """
        self.nLayers, self.ySize, self.xSize = capacity_matrix.shape
        self.capacity_matrix = capacity_matrix.copy()
        self.original_capacity_matrix = capacity_matrix.copy()
        self.layer_directions = layer_directions
        
        # Track routing state
        self.occupancy_matrix = np.zeros_like(capacity_matrix, dtype=np.int16)
        self.routed_nets = []  # List of routed net paths
        self.current_net_path = []  # Current net being routed
        
        # Graph representations
        self.rx_graph = None  # Rustworkx graph (fast!)
        self.node_to_idx = {}  # Map (layer, y, x) -> graph node index
        self.idx_to_node = {}  # Map graph node index -> (layer, y, x)
        
        # Build initial graph
        self._build_graph()
    
    def _build_graph(self):
        """Build Rustworkx graph representation of the routing grid"""
        if not RUSTWORKX_AVAILABLE:
            return
            
        self.rx_graph = rx.PyDiGraph()
        
        # Add nodes for each GCell
        node_idx = 0
        for layer in range(self.nLayers):
            for y in range(self.ySize):
                for x in range(self.xSize):
                    node_id = (layer, y, x)
                    capacity = self.capacity_matrix[layer, y, x]
                    occupancy = self.occupancy_matrix[layer, y, x]
                    congestion = self._calculate_congestion(layer, y, x)
                    
                    # Add node with data
                    node_data = {
                        'layer': layer,
                        'x': x,
                        'y': y,
                        'capacity': capacity,
                        'original_capacity': self.original_capacity_matrix[layer, y, x],
                        'congestion': congestion,
                        'is_blocked': (capacity == 0),
                        'occupancy': occupancy,
                    }
                    
                    idx = self.rx_graph.add_node(node_data)
                    self.node_to_idx[node_id] = idx
                    self.idx_to_node[idx] = node_id
                    node_idx += 1
        
        # Add edges based on layer directions
        self._add_routing_edges()
    
    def _add_routing_edges(self):
        """Add edges between GCells based on routing rules"""
        # Skip layer 0 (typically not routable)
        for layer in range(1, self.nLayers):
            direction = self.layer_directions[layer]
            
            # Horizontal connections (x-direction)
            if direction == 0:
                for y in range(self.ySize):
                    for x in range(self.xSize - 1):
                        node1 = (layer, y, x)
                        node2 = (layer, y, x + 1)
                        self._add_bidirectional_edge(node1, node2, 'horizontal')
            
            # Vertical connections (y-direction)
            elif direction == 1:
                for y in range(self.ySize - 1):
                    for x in range(self.xSize):
                        node1 = (layer, y, x)
                        node2 = (layer, y + 1, x)
                        self._add_bidirectional_edge(node1, node2, 'vertical')
        
        # Add via connections between layers
        for layer in range(self.nLayers - 1):
            for y in range(self.ySize):
                for x in range(self.xSize):
                    node1 = (layer, y, x)
                    node2 = (layer + 1, y, x)
                    self._add_bidirectional_edge(node1, node2, 'via')
    
    def _add_bidirectional_edge(self, node1: Tuple, node2: Tuple, edge_type: str):
        """Add bidirectional edge with routing constraints"""
        if not RUSTWORKX_AVAILABLE:
            return
            
        cap1 = self.capacity_matrix[node1]
        cap2 = self.capacity_matrix[node2]
        
        # Only add edge if both nodes have capacity
        if cap1 > 0 and cap2 > 0:
            # Calculate edge cost based on congestion
            cost = self._calculate_edge_cost(node1, node2, edge_type)
            
            edge_data = {
                'edge_type': edge_type,
                'cost': cost,
                'capacity': min(cap1, cap2),
            }
            
            idx1 = self.node_to_idx[node1]
            idx2 = self.node_to_idx[node2]
            
            # Add edges (bidirectional)
            self.rx_graph.add_edge(idx1, idx2, edge_data)
            self.rx_graph.add_edge(idx2, idx1, edge_data)
    
    def _calculate_congestion(self, layer: int, y: int, x: int) -> float:
        """Calculate congestion at a GCell"""
        original = self.original_capacity_matrix[layer, y, x]
        if original == 0:
            return 1.0  # Fully blocked
        current = self.capacity_matrix[layer, y, x]
        return 1.0 - (current / original)
    
    def _calculate_edge_cost(self, node1: Tuple, node2: Tuple, edge_type: str) -> float:
        """Calculate edge cost based on congestion and edge type"""
        from config import RouterConfig
        
        # Base cost
        if edge_type == 'via':
            base_cost = RouterConfig.VIA_COST
        else:
            base_cost = 1.0 * RouterConfig.MULTIPLY_COST
        
        # Congestion penalty
        cong1 = self._calculate_congestion(*node1)
        cong2 = self._calculate_congestion(*node2)
        avg_congestion = (cong1 + cong2) / 2.0
        
        # Increase cost with congestion
        congestion_multiplier = 1.0 + (avg_congestion * 10.0)
        
        return base_cost * congestion_multiplier
    
    def get_neighbors(self, node: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        """Get valid neighboring nodes for a given node"""
        if not RUSTWORKX_AVAILABLE or node not in self.node_to_idx:
            return []
        
        idx = self.node_to_idx[node]
        neighbor_indices = self.rx_graph.successor_indices(idx)
        return [self.idx_to_node[n_idx] for n_idx in neighbor_indices]
    
    def get_valid_actions(self, current_node: Tuple[int, int, int]) -> List[str]:
        """
        Get valid actions from current node
        Returns: List of actions ['L', 'R', 'U', 'D', 'UP_LAYER', 'DOWN_LAYER']
        """
        layer, y, x = current_node
        actions = []
        
        # Check layer routing direction
        if layer > 0:
            direction = self.layer_directions[layer]
            
            # Horizontal movement
            if direction == 0:
                if x > 0 and self.capacity_matrix[layer, y, x - 1] > 0:
                    actions.append('L')  # Left
                if x < self.xSize - 1 and self.capacity_matrix[layer, y, x + 1] > 0:
                    actions.append('R')  # Right
            
            # Vertical movement
            elif direction == 1:
                if y > 0 and self.capacity_matrix[layer, y - 1, x] > 0:
                    actions.append('D')  # Down
                if y < self.ySize - 1 and self.capacity_matrix[layer, y + 1, x] > 0:
                    actions.append('U')  # Up
        
        # Via movements
        if layer > 0 and self.capacity_matrix[layer - 1, y, x] > 0:
            actions.append('DOWN_LAYER')
        if layer < self.nLayers - 1 and self.capacity_matrix[layer + 1, y, x] > 0:
            actions.append('UP_LAYER')
        
        return actions
    
    def apply_action(self, current_node: Tuple[int, int, int], action: str) -> Optional[Tuple[int, int, int]]:
        """
        Apply an action and return the next node
        Returns None if action is invalid
        """
        layer, y, x = current_node
        
        next_node = None
        if action == 'L':
            next_node = (layer, y, x - 1)
        elif action == 'R':
            next_node = (layer, y, x + 1)
        elif action == 'U':
            next_node = (layer, y + 1, x)
        elif action == 'D':
            next_node = (layer, y - 1, x)
        elif action == 'UP_LAYER':
            next_node = (layer + 1, y, x)
        elif action == 'DOWN_LAYER':
            next_node = (layer - 1, y, x)
        
        # Validate next node
        if next_node:
            l, y_new, x_new = next_node
            if (0 <= l < self.nLayers and 
                0 <= y_new < self.ySize and 
                0 <= x_new < self.xSize and
                self.capacity_matrix[l, y_new, x_new] > 0):
                return next_node
        
        return None
    
    def update_with_route(self, path: List[Tuple[int, int, int]], net_name: str = ""):
        """
        Update state after routing a net
        Decreases capacity and updates congestion
        """
        if len(path) < 2:
            return
        
        # Update capacity and occupancy for nodes
        for node in path:
            layer, y, x = node
            self.capacity_matrix[layer, y, x] -= 1
            self.occupancy_matrix[layer, y, x] += 1
        
        # Store routed net
        self.routed_nets.append({
            'name': net_name,
            'path': path,
        })
        
        # Update graph edges with new costs
        self._update_edge_costs()
    
    def _update_edge_costs(self):
        """Update all edge costs based on current congestion"""
        if not RUSTWORKX_AVAILABLE:
            return
            
        # Update edge weights in rustworkx graph
        for edge in self.rx_graph.edge_list():
            idx1, idx2 = edge
            node1 = self.idx_to_node[idx1]
            node2 = self.idx_to_node[idx2]
            
            edge_data = self.rx_graph.get_edge_data(idx1, idx2)
            edge_type = edge_data['edge_type']
            new_cost = self._calculate_edge_cost(node1, node2, edge_type)
            
            # Update edge data
            edge_data['cost'] = new_cost
            self.rx_graph.update_edge(idx1, idx2, edge_data)
    
    def to_pyg_data(self, source: Tuple, target: Tuple) -> 'Data':
        """
        Convert to PyTorch Geometric Data for GraphSAGE
        Simple homogeneous graph - much faster!
        """
        if not PYTORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("PyTorch Geometric is required for GNN features")
        if not RUSTWORKX_AVAILABLE:
            raise ImportError("Rustworkx is required for graph operations")
        
        # Create node features for each layer
        layer_nodes = {i: [] for i in range(self.nLayers)}
        node_to_idx = {}
        
        idx = 0
        for node in self.nx_graph.nodes():
            layer, y, x = node
            
            # Mark source and target
            is_source = (node == source)
            is_target = (node == target)
            
            gcell = GCellNode(
                layer=layer,
                x=x,
                y=y,
                capacity=self.capacity_matrix[layer, y, x],
                original_capacity=self.original_capacity_matrix[layer, y, x],
                congestion=self._calculate_congestion(layer, y, x),
                is_blocked=(self.capacity_matrix[layer, y, x] == 0),
                is_terminal=is_source or is_target,
                is_source=is_source,
                is_target=is_target,
                occupancy=self.occupancy_matrix[layer, y, x],
            )
            
            layer_nodes[layer].append(gcell.to_feature_vector())
            node_to_idx[node] = (layer, len(layer_nodes[layer]) - 1)
            idx += 1
        
        # Add node features for each layer
        for layer in range(self.nLayers):
            if layer_nodes[layer]:
                node_features = np.stack(layer_nodes[layer])
                hetero_data[f'layer_{layer}'].x = torch.tensor(node_features, dtype=torch.float)
        
        # Add edges with their types
        edge_types = {'horizontal': [], 'vertical': [], 'via': []}
        
        for node1, node2, data in self.nx_graph.edges(data=True):
            edge_type = data['edge_type']
            layer1, idx1 = node_to_idx[node1]
            layer2, idx2 = node_to_idx[node2]
            
            if edge_type in ['horizontal', 'vertical']:
                edge_types[edge_type].append([idx1, idx2])
            elif edge_type == 'via':
                edge_types['via'].append([idx1, idx2])
        
        # Add edge indices
        for edge_type, edges in edge_types.items():
            if edges:
                edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
                if edge_type == 'via':
                    # Vias connect different layers
                    pass  # Handled separately in heterogeneous graphs
                else:
                    # Same layer connections
                    pass
        
        return hetero_data
    
    def to_feature_matrix(self, source: Tuple, target: Tuple) -> np.ndarray:
        """
        Convert state to a feature matrix for simple RL agents
        Returns: Feature matrix [layers, y, x, features]
        """
        num_features = 7
        features = np.zeros((self.nLayers, self.ySize, self.xSize, num_features), dtype=np.float32)
        
        for layer in range(self.nLayers):
            for y in range(self.ySize):
                for x in range(self.xSize):
                    features[layer, y, x, 0] = self.capacity_matrix[layer, y, x]
                    features[layer, y, x, 1] = self._calculate_congestion(layer, y, x)
                    features[layer, y, x, 2] = self.occupancy_matrix[layer, y, x]
                    features[layer, y, x, 3] = float((layer, y, x) == source)
                    features[layer, y, x, 4] = float((layer, y, x) == target)
                    features[layer, y, x, 5] = abs(y - target[1]) + abs(x - target[2])  # Manhattan distance
                    features[layer, y, x, 6] = float(self.capacity_matrix[layer, y, x] == 0)
        
        return features
    
    def get_state_summary(self) -> Dict:
        """Get summary statistics of current state"""
        return {
            'total_capacity': self.capacity_matrix.sum(),
            'original_capacity': self.original_capacity_matrix.sum(),
            'utilization': 1.0 - (self.capacity_matrix.sum() / max(self.original_capacity_matrix.sum(), 1)),
            'num_routed_nets': len(self.routed_nets),
            'avg_congestion': np.mean([self._calculate_congestion(l, y, x) 
                                       for l in range(self.nLayers)
                                       for y in range(self.ySize)
                                       for x in range(self.xSize)
                                       if self.original_capacity_matrix[l, y, x] > 0]),
            'blocked_cells': np.sum(self.capacity_matrix == 0),
            'num_nodes': self.rx_graph.num_nodes() if RUSTWORKX_AVAILABLE and self.rx_graph else 0,
            'num_edges': self.rx_graph.num_edges() if RUSTWORKX_AVAILABLE and self.rx_graph else 0,
        }
    
    def visualize_layer(self, layer: int, current_pos: Optional[Tuple] = None,
                       target: Optional[Tuple] = None) -> str:
        """Create ASCII visualization of a layer"""
        vis = []
        vis.append(f"=== Layer {layer} ===")
        
        for y in range(self.ySize):
            row = []
            for x in range(self.xSize):
                if current_pos and current_pos == (layer, y, x):
                    row.append('C')  # Current position
                elif target and target == (layer, y, x):
                    row.append('T')  # Target
                elif self.capacity_matrix[layer, y, x] == 0:
                    row.append('#')  # Blocked
                elif self.occupancy_matrix[layer, y, x] > 0:
                    row.append('*')  # Occupied
                else:
                    row.append('.')  # Free
            vis.append(' '.join(row))
        
        return '\n'.join(vis)
