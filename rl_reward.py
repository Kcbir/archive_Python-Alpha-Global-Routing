# coding: utf-8
"""
Reward Formulation for VLSI Routing Reinforcement Learning

Implements the reward structure:
1. Large positive: reached target
2. Large negative: dead end
3. Manhattan distance change (moving toward/away from target)
4. Small negative for each action (effort cost)
5. Congestion penalty
6. Direction change penalty (leads to vias)
"""

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class RewardConfig:
    """Configuration for reward calculation"""
    # Terminal rewards
    REWARD_REACH_TARGET = 1000.0          # Large positive for reaching target
    REWARD_DEAD_END = -1000.0             # Large negative for dead end
    
    # Distance-based rewards
    REWARD_CLOSER_TO_TARGET = 10.0        # Moving toward target
    REWARD_AWAY_FROM_TARGET = -15.0       # Moving away (detour penalty)
    
    # Action costs
    COST_PER_STEP = -1.0                  # Small negative for each action
    COST_DIRECTION_CHANGE = -5.0          # Penalty for changing direction
    COST_VIA = -10.0                      # Penalty for using via (layer change)
    
    # Congestion rewards
    REWARD_LOW_CONGESTION = 5.0           # Bonus for using low congestion paths
    PENALTY_HIGH_CONGESTION = -20.0       # Penalty for high congestion areas
    CONGESTION_THRESHOLD_LOW = 0.3        # Below this is "low congestion"
    CONGESTION_THRESHOLD_HIGH = 0.7       # Above this is "high congestion"
    
    # Path quality
    PENALTY_REVISIT = -50.0               # Penalty for revisiting a node
    REWARD_STRAIGHT_PATH = 2.0            # Bonus for maintaining direction
    
    # Final path rewards (when target is reached)
    BONUS_SHORT_PATH = 100.0              # Bonus for efficient path
    PENALTY_LONG_PATH = 50.0              # Penalty for inefficient path
    PATH_EFFICIENCY_THRESHOLD = 1.5       # Path length / Manhattan distance
    

class RewardCalculator:
    """Calculate rewards for routing actions in RL framework"""
    
    def __init__(self, config: Optional[RewardConfig] = None):
        """
        Initialize reward calculator
        
        Args:
            config: RewardConfig object with reward parameters
        """
        self.config = config if config else RewardConfig()
        self.step_count = 0
        self.direction_history = []
        self.visited_nodes = set()
        
    def reset(self):
        """Reset calculator for new episode/net"""
        self.step_count = 0
        self.direction_history = []
        self.visited_nodes = set()
    
    def calculate_reward(self,
                        current_node: Tuple[int, int, int],
                        next_node: Tuple[int, int, int],
                        target_node: Tuple[int, int, int],
                        action: str,
                        state,
                        is_terminal: bool = False,
                        is_dead_end: bool = False) -> Tuple[float, dict]:
        """
        Calculate reward for taking an action
        
        Args:
            current_node: Current GCell position (layer, y, x)
            next_node: Next GCell position after action
            target_node: Target GCell position
            action: Action taken ('L', 'R', 'U', 'D', 'UP_LAYER', 'DOWN_LAYER')
            state: RoutingGraphState object
            is_terminal: Whether reached target
            is_dead_end: Whether reached dead end
            
        Returns:
            reward: Total reward value
            reward_breakdown: Dictionary with reward components
        """
        reward = 0.0
        breakdown = {}
        
        # 1. Terminal conditions
        if is_terminal:
            reward += self.config.REWARD_REACH_TARGET
            breakdown['target_reached'] = self.config.REWARD_REACH_TARGET
            
            # Add path efficiency bonus/penalty
            path_efficiency = self._calculate_path_efficiency(
                self.step_count, current_node, target_node
            )
            if path_efficiency < self.config.PATH_EFFICIENCY_THRESHOLD:
                bonus = self.config.BONUS_SHORT_PATH
                reward += bonus
                breakdown['efficiency_bonus'] = bonus
            else:
                penalty = -self.config.PENALTY_LONG_PATH * (path_efficiency - 1.0)
                reward += penalty
                breakdown['efficiency_penalty'] = penalty
        
        elif is_dead_end:
            reward += self.config.REWARD_DEAD_END
            breakdown['dead_end'] = self.config.REWARD_DEAD_END
        
        # 2. Distance-based reward (Manhattan distance change)
        prev_distance = self._manhattan_distance(current_node, target_node)
        new_distance = self._manhattan_distance(next_node, target_node)
        distance_change = prev_distance - new_distance
        
        if distance_change > 0:
            # Moving closer
            distance_reward = self.config.REWARD_CLOSER_TO_TARGET * distance_change
            breakdown['closer_to_target'] = distance_reward
        elif distance_change < 0:
            # Moving away (detour)
            distance_reward = self.config.REWARD_AWAY_FROM_TARGET * abs(distance_change)
            breakdown['detour_penalty'] = distance_reward
        else:
            # Same distance
            distance_reward = 0.0
        
        reward += distance_reward
        
        # 3. Step cost (incremental effort)
        reward += self.config.COST_PER_STEP
        breakdown['step_cost'] = self.config.COST_PER_STEP
        
        # 4. Direction change penalty
        direction = self._get_direction_from_action(action)
        if len(self.direction_history) > 0 and direction != self.direction_history[-1]:
            # Changed direction
            if self._is_via_change(action):
                # Via change (layer change)
                penalty = self.config.COST_VIA
                breakdown['via_cost'] = penalty
            else:
                # Horizontal/vertical direction change
                penalty = self.config.COST_DIRECTION_CHANGE
                breakdown['direction_change'] = penalty
            reward += penalty
        else:
            # Continuing in same direction
            bonus = self.config.REWARD_STRAIGHT_PATH
            reward += bonus
            breakdown['straight_path'] = bonus
        
        self.direction_history.append(direction)
        
        # 5. Congestion-based reward
        congestion = state._calculate_congestion(*next_node)
        
        if congestion < self.config.CONGESTION_THRESHOLD_LOW:
            # Low congestion - good choice
            congestion_reward = self.config.REWARD_LOW_CONGESTION
            breakdown['low_congestion'] = congestion_reward
        elif congestion > self.config.CONGESTION_THRESHOLD_HIGH:
            # High congestion - bad choice
            congestion_reward = self.config.PENALTY_HIGH_CONGESTION * congestion
            breakdown['high_congestion'] = congestion_reward
        else:
            # Medium congestion
            congestion_reward = -5.0 * congestion
            breakdown['medium_congestion'] = congestion_reward
        
        reward += congestion_reward
        
        # 6. Revisit penalty
        if next_node in self.visited_nodes:
            penalty = self.config.PENALTY_REVISIT
            reward += penalty
            breakdown['revisit_penalty'] = penalty
        
        self.visited_nodes.add(next_node)
        self.step_count += 1
        
        return reward, breakdown
    
    def _manhattan_distance(self, node1: Tuple[int, int, int], 
                           node2: Tuple[int, int, int]) -> int:
        """Calculate Manhattan distance between two nodes (ignoring layer)"""
        layer1, y1, x1 = node1
        layer2, y2, x2 = node2
        return abs(y1 - y2) + abs(x1 - x2)
    
    def _get_direction_from_action(self, action: str) -> str:
        """Map action to direction type"""
        direction_map = {
            'L': 'horizontal',
            'R': 'horizontal',
            'U': 'vertical',
            'D': 'vertical',
            'UP_LAYER': 'via',
            'DOWN_LAYER': 'via',
        }
        return direction_map.get(action, 'unknown')
    
    def _is_via_change(self, action: str) -> bool:
        """Check if action is a layer change (via)"""
        return action in ['UP_LAYER', 'DOWN_LAYER']
    
    def _calculate_path_efficiency(self, path_length: int,
                                   start: Tuple[int, int, int],
                                   target: Tuple[int, int, int]) -> float:
        """
        Calculate path efficiency ratio
        Efficiency = actual_path_length / manhattan_distance
        """
        manhattan = self._manhattan_distance(start, target)
        if manhattan == 0:
            return 1.0
        return path_length / manhattan
    
    def calculate_final_path_reward(self, path: List[Tuple[int, int, int]],
                                    target: Tuple[int, int, int],
                                    state) -> Tuple[float, dict]:
        """
        Calculate final reward for complete path
        Used for offline evaluation or episode-end bonus
        
        Args:
            path: Complete path from source to target
            target: Target node
            state: RoutingGraphState object
            
        Returns:
            reward: Final path reward
            breakdown: Dictionary with reward components
        """
        if len(path) < 2:
            return 0.0, {}
        
        reward = 0.0
        breakdown = {}
        
        # Path length metrics
        path_length = len(path) - 1
        manhattan = self._manhattan_distance(path[0], target)
        efficiency = path_length / max(manhattan, 1)
        
        breakdown['path_length'] = path_length
        breakdown['manhattan_distance'] = manhattan
        breakdown['efficiency_ratio'] = efficiency
        
        # Efficiency reward
        if efficiency < self.config.PATH_EFFICIENCY_THRESHOLD:
            bonus = self.config.BONUS_SHORT_PATH * (2.0 - efficiency)
            reward += bonus
            breakdown['efficiency_bonus'] = bonus
        else:
            penalty = -self.config.PENALTY_LONG_PATH * (efficiency - 1.0)
            reward += penalty
            breakdown['efficiency_penalty'] = penalty
        
        # Count direction changes and vias
        num_direction_changes = 0
        num_vias = 0
        prev_direction = None
        
        for i in range(len(path) - 1):
            curr = path[i]
            next_node = path[i + 1]
            
            # Detect via
            if curr[0] != next_node[0]:
                num_vias += 1
            
            # Detect direction change
            direction = self._infer_direction(curr, next_node)
            if prev_direction and direction != prev_direction and direction != 'via':
                num_direction_changes += 1
            prev_direction = direction
        
        breakdown['num_vias'] = num_vias
        breakdown['num_direction_changes'] = num_direction_changes
        
        # Penalties for changes
        via_penalty = self.config.COST_VIA * num_vias
        direction_penalty = self.config.COST_DIRECTION_CHANGE * num_direction_changes
        reward += via_penalty + direction_penalty
        breakdown['via_penalty'] = via_penalty
        breakdown['direction_penalty'] = direction_penalty
        
        # Average congestion along path
        avg_congestion = np.mean([state._calculate_congestion(*node) for node in path])
        congestion_penalty = -10.0 * avg_congestion
        reward += congestion_penalty
        breakdown['avg_congestion'] = avg_congestion
        breakdown['congestion_penalty'] = congestion_penalty
        
        return reward, breakdown
    
    def _infer_direction(self, node1: Tuple[int, int, int],
                        node2: Tuple[int, int, int]) -> str:
        """Infer direction of movement between two nodes"""
        layer1, y1, x1 = node1
        layer2, y2, x2 = node2
        
        if layer1 != layer2:
            return 'via'
        elif x1 != x2:
            return 'horizontal'
        elif y1 != y2:
            return 'vertical'
        else:
            return 'none'


class AdaptiveRewardCalculator(RewardCalculator):
    """
    Advanced reward calculator with adaptive weights
    Adjusts reward weights based on routing progress
    """
    
    def __init__(self, config: Optional[RewardConfig] = None):
        super().__init__(config)
        self.total_nets = 0
        self.routed_nets = 0
        self.global_congestion = 0.0
    
    def update_global_state(self, total_nets: int, routed_nets: int,
                           global_congestion: float):
        """Update global routing state for adaptive rewards"""
        self.total_nets = total_nets
        self.routed_nets = routed_nets
        self.global_congestion = global_congestion
    
    def calculate_reward(self, current_node, next_node, target_node,
                        action, state, is_terminal=False, is_dead_end=False):
        """Calculate reward with adaptive weights"""
        
        # Get base reward
        reward, breakdown = super().calculate_reward(
            current_node, next_node, target_node, action, state,
            is_terminal, is_dead_end
        )
        
        # Adaptive adjustments based on routing progress
        progress = self.routed_nets / max(self.total_nets, 1)
        
        # Early routing: prioritize efficiency
        # Later routing: prioritize congestion avoidance
        if progress < 0.3:
            # Early stage - care more about distance
            distance_multiplier = 1.5
            congestion_multiplier = 0.7
        elif progress < 0.7:
            # Middle stage - balanced
            distance_multiplier = 1.0
            congestion_multiplier = 1.0
        else:
            # Late stage - care more about congestion
            distance_multiplier = 0.7
            congestion_multiplier = 2.0
        
        # Adjust specific components
        if 'closer_to_target' in breakdown:
            adjustment = breakdown['closer_to_target'] * (distance_multiplier - 1.0)
            reward += adjustment
            breakdown['distance_adjustment'] = adjustment
        
        if 'high_congestion' in breakdown:
            adjustment = breakdown['high_congestion'] * (congestion_multiplier - 1.0)
            reward += adjustment
            breakdown['congestion_adjustment'] = adjustment
        
        # Global congestion awareness
        if self.global_congestion > 0.8:
            # System is highly congested - be more careful
            congestion_awareness = -5.0
            reward += congestion_awareness
            breakdown['global_congestion_penalty'] = congestion_awareness
        
        return reward, breakdown
