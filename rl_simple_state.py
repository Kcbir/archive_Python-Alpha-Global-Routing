# coding: utf-8
"""
Simple Graph State using Rustworkx (FAST!)
Minimal, clean implementation for VLSI routing
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

try:
    import rustworkx as rx
    RUSTWORKX_AVAILABLE = True  
except ImportError:
    RUSTWORKX_AVAILABLE = False
    print("Install rustworkx: pip install rustworkx")


@dataclass
class GCell:
    """Simple GCell node"""
    layer: int
    x: int
    y: int
    capacity: float
    congestion: float
    is_source: bool = False
    is_target: bool = False
    
    def features(self) -> np.ndarray:
        """11D feature vector for GNN"""
        return np.array([
            self.layer, self.x, self.y,
            self.capacity, self.congestion,
            float(self.is_source), float(self.is_target),
            0, 0, 0, 0  # padding for compatibility
        ], dtype=np.float32)


class SimpleRoutingState:
    """
    Fast graph-based state using rustworkx
    Simple, clean, and FAST!
    """
    
    def __init__(self, capacity_matrix: np.ndarray, layer_directions: List[int]):
        self.nLayers, self.ySize, self.xSize = capacity_matrix.shape
        self.capacity = capacity_matrix.copy()
        self.original_capacity = capacity_matrix.copy()
        self.layer_dirs = layer_directions
        
        # Rustworkx graph (FAST!)
        self.graph = rx.PyDiGraph() if RUSTWORKX_AVAILABLE else None
        self.node_map = {}  # (layer, y, x) -> graph_idx
        self.idx_map = {}   # graph_idx -> (layer, y, x)
        
        # Build graph
        self._build_graph()
    
    def _build_graph(self):
        """Build graph structure"""
        if not RUSTWORKX_AVAILABLE:
            return
        
        # Add all nodes
        idx = 0
        for l in range(self.nLayers):
            for y in range(self.ySize):
                for x in range(self.xSize):
                    node_id = (l, y, x)
                    capacity = self.capacity[l, y, x]
                    
                    node_data = {
                        'layer': l, 'x': x, 'y': y,
                        'capacity': capacity,
                        'blocked': (capacity == 0)
                    }
                    
                    graph_idx = self.graph.add_node(node_data)
                    self.node_map[node_id] = graph_idx
                    self.idx_map[graph_idx] = node_id
                    idx += 1
        
        # Add edges
        self._add_edges()
    
    def _add_edges(self):
        """Add routing edges"""
        if not RUSTWORKX_AVAILABLE:
            return
        
        for l in range(1, self.nLayers):  # Skip layer 0
            direction = self.layer_dirs[l]
            
            # Horizontal (x-direction)
            if direction == 0:
                for y in range(self.ySize):
                    for x in range(self.xSize - 1):
                        if self.capacity[l, y, x] > 0 and self.capacity[l, y, x+1] > 0:
                            i1 = self.node_map[(l, y, x)]
                            i2 = self.node_map[(l, y, x+1)]
                            self.graph.add_edge(i1, i2, {'type': 'h', 'cost': 1})
                            self.graph.add_edge(i2, i1, {'type': 'h', 'cost': 1})
            
            # Vertical (y-direction)
            elif direction == 1:
                for y in range(self.ySize - 1):
                    for x in range(self.xSize):
                        if self.capacity[l, y, x] > 0 and self.capacity[l, y+1, x] > 0:
                            i1 = self.node_map[(l, y, x)]
                            i2 = self.node_map[(l, y+1, x)]
                            self.graph.add_edge(i1, i2, {'type': 'v', 'cost': 1})
                            self.graph.add_edge(i2, i1, {'type': 'v', 'cost': 1})
        
        # Vias between layers
        for l in range(self.nLayers - 1):
            for y in range(self.ySize):
                for x in range(self.xSize):
                    if self.capacity[l, y, x] > 0 and self.capacity[l+1, y, x] > 0:
                        i1 = self.node_map[(l, y, x)]
                        i2 = self.node_map[(l+1, y, x)]
                        self.graph.add_edge(i1, i2, {'type': 'via', 'cost': 2})
                        self.graph.add_edge(i2, i1, {'type': 'via', 'cost': 2})
    
    def get_valid_actions(self, node: Tuple[int, int, int]) -> List[str]:
        """Get valid actions from a node"""
        layer, y, x = node
        actions = []
        
        if layer > 0:
            direction = self.layer_dirs[layer]
            
            if direction == 0:  # Horizontal
                if x > 0 and self.capacity[layer, y, x-1] > 0:
                    actions.append('L')
                if x < self.xSize - 1 and self.capacity[layer, y, x+1] > 0:
                    actions.append('R')
            elif direction == 1:  # Vertical
                if y > 0 and self.capacity[layer, y-1, x] > 0:
                    actions.append('D')
                if y < self.ySize - 1 and self.capacity[layer, y+1, x] > 0:
                    actions.append('U')
        
        # Vias
        if layer > 0 and self.capacity[layer-1, y, x] > 0:
            actions.append('DOWN_LAYER')
        if layer < self.nLayers - 1 and self.capacity[layer+1, y, x] > 0:
            actions.append('UP_LAYER')
        
        return actions
    
    def apply_action(self, node: Tuple[int, int, int], action: str) -> Optional[Tuple[int, int, int]]:
        """Apply action and return next node"""
        l, y, x = node
        
        moves = {
            'L': (l, y, x-1),
            'R': (l, y, x+1),
            'U': (l, y+1, x),
            'D': (l, y-1, x),
            'UP_LAYER': (l+1, y, x),
            'DOWN_LAYER': (l-1, y, x),
        }
        
        next_node = moves.get(action)
        if next_node:
            nl, ny, nx = next_node
            if (0 <= nl < self.nLayers and 
                0 <= ny < self.ySize and 
                0 <= nx < self.xSize and
                self.capacity[nl, ny, nx] > 0):
                return next_node
        
        return None
    
    def update_after_routing(self, path: List[Tuple[int, int, int]]):
        """Update state after routing a net"""
        for node in path:
            l, y, x = node
            self.capacity[l, y, x] = max(0, self.capacity[l, y, x] - 1)
    
    def to_numpy_features(self, source: Tuple, target: Tuple) -> np.ndarray:
        """Convert to feature matrix for simple RL"""
        features = np.zeros((self.nLayers, self.ySize, self.xSize, 5), dtype=np.float32)
        
        for l in range(self.nLayers):
            for y in range(self.ySize):
                for x in range(self.xSize):
                    features[l, y, x, 0] = self.capacity[l, y, x] / max(self.original_capacity[l, y, x], 1)
                    features[l, y, x, 1] = float((l, y, x) == source)
                    features[l, y, x, 2] = float((l, y, x) == target)
                    features[l, y, x, 3] = abs(y - target[1]) + abs(x - target[2])  # Manhattan to target
                    features[l, y, x, 4] = float(self.capacity[l, y, x] == 0)
        
        return features
    
    def to_pyg_data(self, source: Tuple, target: Tuple):
        """Convert to PyTorch Geometric Data for GraphSAGE"""
        try:
            import torch
            from torch_geometric.data import Data
        except ImportError:
            print("Need: pip install torch torch-geometric")
            return None
        
        if not RUSTWORKX_AVAILABLE:
            return None
        
        # Node features
        node_features = []
        for idx in range(self.graph.num_nodes()):
            node_id = self.idx_map[idx]
            l, y, x = node_id
            
            is_source = (node_id == source)
            is_target = (node_id == target)
            
            feat = np.array([
                l, x, y,
                self.capacity[l, y, x],
                self.capacity[l, y, x] / max(self.original_capacity[l, y, x], 1),  # congestion
                float(is_source),
                float(is_target),
                abs(y - target[1]) + abs(x - target[2]),  # distance to target
            ], dtype=np.float32)
            
            node_features.append(feat)
        
        # Edge index
        edge_list = self.graph.edge_list()
        edge_index = [[e[0] for e in edge_list], [e[1] for e in edge_list]]
        
        data = Data(
            x=torch.tensor(np.array(node_features), dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            num_nodes=self.graph.num_nodes()
        )
        
        return data
    
    def summary(self) -> Dict:
        """State summary"""
        return {
            'capacity_used': float(self.original_capacity.sum() - self.capacity.sum()),
            'capacity_total': float(self.original_capacity.sum()),
            'utilization': 1.0 - (self.capacity.sum() / max(self.original_capacity.sum(), 1)),
            'num_nodes': self.graph.num_nodes() if RUSTWORKX_AVAILABLE else 0,
            'num_edges': self.graph.num_edges() if RUSTWORKX_AVAILABLE else 0,
        }
