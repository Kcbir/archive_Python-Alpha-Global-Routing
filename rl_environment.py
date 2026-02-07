# coding: utf-8
"""
Reinforcement Learning Environment for VLSI Routing

Provides OpenAI Gym-compatible environment for routing nets in VLSI circuits.
Integrates graph-based state representation and reward calculation.
"""

import numpy as np
import gym
from gym import spaces
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import random

from rl_graph_state import RoutingGraphState
from rl_reward import RewardCalculator, RewardConfig, AdaptiveRewardCalculator


class VLSIRoutingEnv(gym.Env):
    """
    OpenAI Gym Environment for VLSI Routing
    
    State: Graph-based representation of routing grid with GCells
    Action: Move in direction (L, R, U, D, UP_LAYER, DOWN_LAYER)
    Reward: Based on distance to target, congestion, direction changes, etc.
    """
    
    metadata = {'render.modes': ['human', 'ascii']}
    
    def __init__(self,
                 capacity_matrix: np.ndarray,
                 layer_directions: List[int],
                 nets: Dict[str, List[Tuple]],
                 reward_config: Optional[RewardConfig] = None,
                 use_adaptive_rewards: bool = False,
                 max_steps_per_net: int = 1000):
        """
        Initialize VLSI Routing Environment
        
        Args:
            capacity_matrix: 3D array [layers, y, x] with capacity values
            layer_directions: List indicating routing direction per layer
            nets: Dictionary mapping net names to terminal locations
            reward_config: Configuration for reward calculation
            use_adaptive_rewards: Whether to use adaptive reward calculator
            max_steps_per_net: Maximum steps allowed per net routing
        """
        super(VLSIRoutingEnv, self).__init__()
        
        self.capacity_matrix = capacity_matrix
        self.layer_directions = layer_directions
        self.nets = nets
        self.net_names = list(nets.keys())
        self.max_steps_per_net = max_steps_per_net
        
        # Initialize state and reward
        self.state = RoutingGraphState(capacity_matrix, layer_directions)
        
        if use_adaptive_rewards:
            self.reward_calculator = AdaptiveRewardCalculator(reward_config)
        else:
            self.reward_calculator = RewardCalculator(reward_config)
        
        # Action space: 6 discrete actions
        # 0: Left, 1: Right, 2: Up, 3: Down, 4: UP_LAYER, 5: DOWN_LAYER
        self.action_space = spaces.Discrete(6)
        self.action_to_str = {
            0: 'L',
            1: 'R', 
            2: 'U',
            3: 'D',
            4: 'UP_LAYER',
            5: 'DOWN_LAYER',
        }
        
        # Observation space: feature matrix
        # Shape: [layers, y, x, features]
        nLayers, ySize, xSize = capacity_matrix.shape
        self.observation_space = spaces.Box(
            low=0,
            high=1000,
            shape=(nLayers, ySize, xSize, 7),
            dtype=np.float32
        )
        
        # Episode tracking
        self.current_net_idx = 0
        self.current_net_name = None
        self.current_terminals = None
        self.source = None
        self.target = None
        self.current_position = None
        self.current_path = []
        self.step_count = 0
        self.episode_rewards = []
        self.done = False
        
    def reset(self, net_idx: Optional[int] = None) -> np.ndarray:
        """
        Reset environment for a new net
        
        Args:
            net_idx: Index of net to route (None for sequential)
            
        Returns:
            observation: Initial state observation
        """
        # Select net to route
        if net_idx is not None:
            self.current_net_idx = net_idx
        
        if self.current_net_idx >= len(self.net_names):
            # All nets routed - reset completely
            self.state = RoutingGraphState(self.capacity_matrix, self.layer_directions)
            self.current_net_idx = 0
        
        self.current_net_name = self.net_names[self.current_net_idx]
        self.current_terminals = self.nets[self.current_net_name]
        
        # Select source and target (for multi-terminal nets, use Steiner approach later)
        # For now, connect first two terminals
        if len(self.current_terminals) < 2:
            # Single terminal net - skip
            return self._get_observation()
        
        # Extract source and target from terminal format
        # Format: [(layer, x, y, metal_name), ...]
        source_terminal = self.current_terminals[0]
        target_terminal = self.current_terminals[1]
        
        self.source = (source_terminal[0], source_terminal[2], source_terminal[1])  # (layer, y, x)
        self.target = (target_terminal[0], target_terminal[2], target_terminal[1])  # (layer, y, x)
        
        self.current_position = self.source
        self.current_path = [self.current_position]
        self.step_count = 0
        self.done = False
        
        # Reset reward calculator
        self.reward_calculator.reset()
        
        # Update adaptive rewards if enabled
        if isinstance(self.reward_calculator, AdaptiveRewardCalculator):
            global_congestion = self.state.get_state_summary()['avg_congestion']
            self.reward_calculator.update_global_state(
                total_nets=len(self.net_names),
                routed_nets=self.current_net_idx,
                global_congestion=global_congestion
            )
        
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute one step in the environment
        
        Args:
            action: Action index (0-5)
            
        Returns:
            observation: Next state observation
            reward: Reward for this step
            done: Whether episode is complete
            info: Additional information
        """
        if self.done:
            raise RuntimeError("Episode is done. Call reset() to start new episode.")
        
        # Convert action to string
        action_str = self.action_to_str[action]
        
        # Check if action is valid
        valid_actions = self.state.get_valid_actions(self.current_position)
        
        info = {
            'valid_actions': valid_actions,
            'action_taken': action_str,
            'current_position': self.current_position,
            'target_position': self.target,
        }
        
        # Check for dead end or invalid action
        if action_str not in valid_actions:
            # Invalid action - dead end or blocked
            reward, breakdown = self.reward_calculator.calculate_reward(
                current_node=self.current_position,
                next_node=self.current_position,
                target_node=self.target,
                action=action_str,
                state=self.state,
                is_terminal=False,
                is_dead_end=True
            )
            self.done = True
            info['terminal_reason'] = 'invalid_action'
            info['reward_breakdown'] = breakdown
            return self._get_observation(), reward, self.done, info
        
        # Apply action
        next_position = self.state.apply_action(self.current_position, action_str)
        
        if next_position is None:
            # Should not happen if action was valid, but handle gracefully
            reward = -100.0
            self.done = True
            info['terminal_reason'] = 'invalid_state'
            return self._get_observation(), reward, self.done, info
        
        # Check if reached target
        is_terminal = (next_position == self.target)
        
        # Calculate reward
        reward, breakdown = self.reward_calculator.calculate_reward(
            current_node=self.current_position,
            next_node=next_position,
            target_node=self.target,
            action=action_str,
            state=self.state,
            is_terminal=is_terminal,
            is_dead_end=False
        )
        
        info['reward_breakdown'] = breakdown
        
        # Update position and path
        self.current_position = next_position
        self.current_path.append(next_position)
        self.step_count += 1
        
        # Check termination conditions
        if is_terminal:
            # Reached target - success!
            self.done = True
            info['terminal_reason'] = 'target_reached'
            info['path_length'] = len(self.current_path)
            
            # Update state with routed path
            self.state.update_with_route(self.current_path, self.current_net_name)
            
            # Calculate final path reward
            final_reward, final_breakdown = self.reward_calculator.calculate_final_path_reward(
                self.current_path, self.target, self.state
            )
            info['final_reward_breakdown'] = final_breakdown
            reward += final_reward * 0.1  # Add 10% of final reward to current reward
            
        elif self.step_count >= self.max_steps_per_net:
            # Max steps exceeded
            self.done = True
            reward -= 500.0  # Large penalty for timeout
            info['terminal_reason'] = 'max_steps_exceeded'
        
        self.episode_rewards.append(reward)
        info['cumulative_reward'] = sum(self.episode_rewards)
        info['step_count'] = self.step_count
        
        return self._get_observation(), reward, self.done, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current state observation"""
        return self.state.to_feature_matrix(self.source, self.target)
    
    def get_graph_observation(self) -> Any:
        """Get graph-based observation (for GNN agents)"""
        if hasattr(self, 'source') and hasattr(self, 'target'):
            try:
                return self.state.to_pyg_hetero_data(self.source, self.target)
            except ImportError:
                return None
        return None
    
    def render(self, mode: str = 'human') -> Optional[str]:
        """
        Render the environment
        
        Args:
            mode: Rendering mode ('human' or 'ascii')
            
        Returns:
            visualization string if mode='ascii'
        """
        if mode == 'ascii':
            vis = []
            vis.append(f"\n=== Net: {self.current_net_name} ===")
            vis.append(f"Step: {self.step_count}")
            vis.append(f"Position: {self.current_position}")
            vis.append(f"Target: {self.target}")
            vis.append(f"Distance to target: {self._manhattan_distance(self.current_position, self.target)}")
            
            # Visualize current layer
            if self.current_position:
                layer = self.current_position[0]
                vis.append(self.state.visualize_layer(layer, self.current_position, self.target))
            
            return '\n'.join(vis)
        
        elif mode == 'human':
            print(self.render(mode='ascii'))
            return None
    
    def _manhattan_distance(self, node1: Tuple, node2: Tuple) -> int:
        """Calculate Manhattan distance between nodes"""
        return abs(node1[1] - node2[1]) + abs(node1[2] - node2[2])
    
    def get_state_summary(self) -> Dict:
        """Get summary of current state"""
        summary = self.state.get_state_summary()
        summary.update({
            'current_net': self.current_net_name,
            'current_net_idx': self.current_net_idx,
            'total_nets': len(self.net_names),
            'step_count': self.step_count,
            'path_length': len(self.current_path),
        })
        return summary
    
    def route_all_nets_sequentially(self, agent, verbose: bool = False) -> Dict:
        """
        Route all nets sequentially using an agent
        
        Args:
            agent: Agent with select_action(observation, valid_actions) method
            verbose: Whether to print progress
            
        Returns:
            results: Dictionary with routing statistics
        """
        results = {
            'routed_nets': [],
            'failed_nets': [],
            'total_reward': 0.0,
            'avg_path_length': 0.0,
            'routing_success_rate': 0.0,
        }
        
        for net_idx in range(len(self.net_names)):
            obs = self.reset(net_idx)
            done = False
            net_reward = 0.0
            
            while not done:
                # Get valid actions
                valid_actions = self.state.get_valid_actions(self.current_position)
                
                if not valid_actions:
                    # No valid actions - dead end
                    if verbose:
                        print(f"Dead end for net {self.current_net_name}")
                    results['failed_nets'].append(self.current_net_name)
                    break
                
                # Select action using agent
                action = agent.select_action(obs, valid_actions)
                obs, reward, done, info = self.step(action)
                net_reward += reward
                
                if verbose and done:
                    print(f"Net {self.current_net_name}: {info.get('terminal_reason')}, "
                          f"Path length: {info.get('path_length', 0)}, "
                          f"Reward: {net_reward:.2f}")
            
            if done and info.get('terminal_reason') == 'target_reached':
                results['routed_nets'].append({
                    'name': self.current_net_name,
                    'path_length': info.get('path_length'),
                    'reward': net_reward,
                })
            
            results['total_reward'] += net_reward
            self.current_net_idx += 1
        
        # Calculate statistics
        if results['routed_nets']:
            results['avg_path_length'] = np.mean([n['path_length'] 
                                                   for n in results['routed_nets']])
        
        total = len(results['routed_nets']) + len(results['failed_nets'])
        if total > 0:
            results['routing_success_rate'] = len(results['routed_nets']) / total
        
        return results


class RandomAgent:
    """Simple random agent for testing"""
    
    def select_action(self, observation, valid_actions):
        """Select random valid action"""
        # Map valid action strings to action indices
        action_map = {
            'L': 0,
            'R': 1,
            'U': 2,
            'D': 3,
            'UP_LAYER': 4,
            'DOWN_LAYER': 5,
        }
        
        valid_action_indices = [action_map[a] for a in valid_actions if a in action_map]
        
        if valid_action_indices:
            return random.choice(valid_action_indices)
        return 0


class GreedyAgent:
    """Greedy agent that always moves closer to target"""
    
    def __init__(self, target):
        self.target = target
    
    def select_action(self, observation, valid_actions):
        """Select action that minimizes Manhattan distance to target"""
        action_map = {
            'L': 0,
            'R': 1,
            'U': 2,
            'D': 3,
            'UP_LAYER': 4,
            'DOWN_LAYER': 5,
        }
        
        # Heuristic: prefer actions that move toward target
        # This is a simple version - a better one would calculate actual distances
        
        # Priority order based on target direction
        # This is simplified - real implementation would need current position
        
        priority = ['L', 'R', 'U', 'D', 'UP_LAYER', 'DOWN_LAYER']
        
        for action_str in priority:
            if action_str in valid_actions:
                return action_map[action_str]
        
        # Fallback to first valid action
        if valid_actions:
            return action_map[valid_actions[0]]
        
        return 0


def create_vlsi_env_from_files(cap_file: str, net_file: str, **kwargs) -> VLSIRoutingEnv:
    """
    Create VLSI routing environment from .cap and .net files
    
    Args:
        cap_file: Path to .cap file
        net_file: Path to .net file
        **kwargs: Additional arguments for VLSIRoutingEnv
        
    Returns:
        env: VLSIRoutingEnv instance
    """
    from utils import read_cap, read_net
    
    # Read data
    cap_data = read_cap(cap_file, verbose=False)
    net_data = read_net(net_file, verbose=False)
    
    # Extract required information
    capacity_matrix = cap_data['cap']
    layer_directions = cap_data['layerDirections']
    
    # Create environment
    env = VLSIRoutingEnv(
        capacity_matrix=capacity_matrix,
        layer_directions=layer_directions,
        nets=net_data,
        **kwargs
    )
    
    return env
