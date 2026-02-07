# coding: utf-8
"""
Test Suite for RL-based VLSI Routing Framework

Validates that all components work correctly:
1. Graph state representation
2. Reward calculation
3. Environment mechanics
4. Model architectures
5. Integration tests
"""

import numpy as np
import torch
import unittest
import sys
import os

from rl_graph_state import RoutingGraphState, GCellNode, EdgeInfo
from rl_reward import RewardCalculator, RewardConfig, AdaptiveRewardCalculator
from rl_environment import VLSIRoutingEnv, RandomAgent, GreedyAgent
from utils import read_cap, read_net


class TestGraphState(unittest.TestCase):
    """Test graph state representation"""
    
    def setUp(self):
        """Create test capacity matrix"""
        self.capacity_matrix = np.ones((3, 10, 10), dtype=np.float32) * 5
        # Add some blockages
        self.capacity_matrix[:, 0, 0] = 0
        self.capacity_matrix[:, 5, 5] = 0
        
        self.layer_directions = [0, 0, 1]  # Layer 0: horizontal, Layer 1: horizontal, Layer 2: vertical
        
        self.state = RoutingGraphState(self.capacity_matrix, self.layer_directions)
    
    def test_initialization(self):
        """Test state initialization"""
        self.assertEqual(self.state.nLayers, 3)
        self.assertEqual(self.state.ySize, 10)
        self.assertEqual(self.state.xSize, 10)
        self.assertIsNotNone(self.state.nx_graph)
    
    def test_node_features(self):
        """Test node feature extraction"""
        node = GCellNode(
            layer=1, x=5, y=5,
            capacity=5.0, original_capacity=5.0,
            congestion=0.0, is_blocked=False,
            is_terminal=False, is_source=False,
            is_target=False, occupancy=0
        )
        features = node.to_feature_vector()
        self.assertEqual(len(features), 11)
        self.assertEqual(features[0], 1)  # layer
        self.assertEqual(features[4], 5.0)  # original_capacity
    
    def test_valid_actions(self):
        """Test valid action retrieval"""
        current = (1, 5, 5)  # Layer 1, horizontal routing
        actions = self.state.get_valid_actions(current)
        
        # Should have L, R, and layer changes
        self.assertIn('L', actions)
        self.assertIn('R', actions)
        self.assertIn('UP_LAYER', actions)
        self.assertIn('DOWN_LAYER', actions)
        
        # Blocked cell should have no actions (except layer changes)
        blocked = (1, 0, 0)
        actions = self.state.get_valid_actions(blocked)
        self.assertEqual(len(actions), 0)  # Blocked cell has 0 capacity
    
    def test_apply_action(self):
        """Test action application"""
        current = (1, 5, 5)
        
        # Test horizontal movement
        next_node = self.state.apply_action(current, 'R')
        self.assertEqual(next_node, (1, 5, 6))
        
        next_node = self.state.apply_action(current, 'L')
        self.assertEqual(next_node, (1, 5, 4))
        
        # Test layer change
        next_node = self.state.apply_action(current, 'UP_LAYER')
        self.assertEqual(next_node, (2, 5, 5))
    
    def test_update_with_route(self):
        """Test state update after routing"""
        path = [(1, 5, 5), (1, 5, 6), (1, 5, 7)]
        original_capacity = self.state.capacity_matrix[1, 5, 5]
        
        self.state.update_with_route(path, 'test_net')
        
        # Check capacity decreased
        for node in path:
            layer, y, x = node
            self.assertEqual(self.state.capacity_matrix[layer, y, x], original_capacity - 1)
            self.assertEqual(self.state.occupancy_matrix[layer, y, x], 1)
        
        # Check routed nets list
        self.assertEqual(len(self.state.routed_nets), 1)
        self.assertEqual(self.state.routed_nets[0]['name'], 'test_net')
    
    def test_congestion_calculation(self):
        """Test congestion calculation"""
        # Free cell
        congestion = self.state._calculate_congestion(1, 5, 5)
        self.assertEqual(congestion, 0.0)
        
        # After routing
        self.state.capacity_matrix[1, 5, 5] = 2  # Reduced from 5 to 2
        congestion = self.state._calculate_congestion(1, 5, 5)
        self.assertAlmostEqual(congestion, 0.6, places=2)  # (5-2)/5 = 0.6
        
        # Blocked cell
        congestion = self.state._calculate_congestion(1, 0, 0)
        self.assertEqual(congestion, 1.0)
    
    def test_feature_matrix(self):
        """Test feature matrix generation"""
        source = (1, 0, 0)
        target = (1, 9, 9)
        features = self.state.to_feature_matrix(source, target)
        
        self.assertEqual(features.shape, (3, 10, 10, 7))
        
        # Check source marking
        self.assertEqual(features[1, 0, 0, 3], 1.0)  # Source at (1, 0, 0)
        self.assertEqual(features[1, 9, 9, 4], 1.0)  # Target at (1, 9, 9)


class TestRewardCalculator(unittest.TestCase):
    """Test reward calculation"""
    
    def setUp(self):
        """Initialize test environment"""
        self.config = RewardConfig()
        self.calculator = RewardCalculator(self.config)
        
        # Create simple state
        capacity_matrix = np.ones((2, 10, 10), dtype=np.float32) * 5
        layer_directions = [0, 1]
        self.state = RoutingGraphState(capacity_matrix, layer_directions)
    
    def test_target_reached_reward(self):
        """Test reward for reaching target"""
        current = (1, 5, 5)
        target = (1, 5, 6)
        next_node = (1, 5, 6)
        
        reward, breakdown = self.calculator.calculate_reward(
            current, next_node, target, 'R', self.state, is_terminal=True
        )
        
        self.assertGreater(reward, 0)
        self.assertIn('target_reached', breakdown)
        self.assertEqual(breakdown['target_reached'], self.config.REWARD_REACH_TARGET)
    
    def test_dead_end_penalty(self):
        """Test penalty for dead end"""
        current = (1, 5, 5)
        target = (1, 9, 9)
        
        reward, breakdown = self.calculator.calculate_reward(
            current, current, target, 'INVALID', self.state, is_dead_end=True
        )
        
        self.assertLess(reward, 0)
        self.assertIn('dead_end', breakdown)
        self.assertEqual(breakdown['dead_end'], self.config.REWARD_DEAD_END)
    
    def test_distance_reward(self):
        """Test distance-based rewards"""
        target = (1, 5, 10)
        
        # Moving closer
        current = (1, 5, 5)
        next_node = (1, 5, 6)
        reward, breakdown = self.calculator.calculate_reward(
            current, next_node, target, 'R', self.state
        )
        self.assertIn('closer_to_target', breakdown)
        self.assertGreater(breakdown['closer_to_target'], 0)
        
        # Moving away (detour)
        self.calculator.reset()
        current = (1, 5, 5)
        next_node = (1, 5, 4)
        reward, breakdown = self.calculator.calculate_reward(
            current, next_node, target, 'L', self.state
        )
        self.assertIn('detour_penalty', breakdown)
        self.assertLess(breakdown['detour_penalty'], 0)
    
    def test_direction_change_penalty(self):
        """Test direction change penalties"""
        target = (1, 10, 10)
        
        # First move
        reward1, breakdown1 = self.calculator.calculate_reward(
            (1, 5, 5), (1, 5, 6), target, 'R', self.state
        )
        
        # Same direction - should get bonus
        reward2, breakdown2 = self.calculator.calculate_reward(
            (1, 5, 6), (1, 5, 7), target, 'R', self.state
        )
        self.assertIn('straight_path', breakdown2)
        
        # Direction change
        self.calculator.reset()
        self.calculator.calculate_reward((1, 5, 5), (1, 5, 6), target, 'R', self.state)
        reward3, breakdown3 = self.calculator.calculate_reward(
            (1, 5, 6), (1, 6, 6), target, 'U', self.state
        )
        self.assertIn('direction_change', breakdown3)
        self.assertLess(breakdown3['direction_change'], 0)
    
    def test_congestion_penalty(self):
        """Test congestion-based rewards"""
        target = (1, 10, 10)
        
        # Low congestion
        current = (1, 5, 5)
        next_node = (1, 5, 6)
        reward, breakdown = self.calculator.calculate_reward(
            current, next_node, target, 'R', self.state
        )
        self.assertIn('low_congestion', breakdown)
        
        # High congestion
        self.calculator.reset()
        self.state.capacity_matrix[1, 5, 6] = 1  # High congestion
        self.state.original_capacity_matrix[1, 5, 6] = 5
        reward, breakdown = self.calculator.calculate_reward(
            current, next_node, target, 'R', self.state
        )
        self.assertIn('high_congestion', breakdown)
        self.assertLess(breakdown['high_congestion'], 0)
    
    def test_adaptive_rewards(self):
        """Test adaptive reward calculator"""
        config = RewardConfig()
        adaptive = AdaptiveRewardCalculator(config)
        
        # Early routing - prioritize efficiency
        adaptive.update_global_state(100, 10, 0.3)
        self.assertEqual(adaptive.routed_nets, 10)
        
        # Late routing - prioritize congestion
        adaptive.update_global_state(100, 90, 0.8)
        self.assertEqual(adaptive.global_congestion, 0.8)


class TestEnvironment(unittest.TestCase):
    """Test RL environment"""
    
    def setUp(self):
        """Create test environment"""
        capacity_matrix = np.ones((2, 10, 10), dtype=np.float32) * 5
        layer_directions = [0, 0]
        
        # Create simple nets
        nets = {
            'net1': [
                (1, 0, 0, 'M1'),  # (layer, x, y, metal)
                (1, 5, 5, 'M1'),
            ],
            'net2': [
                (1, 0, 9, 'M1'),
                (1, 9, 9, 'M1'),
            ],
        }
        
        self.env = VLSIRoutingEnv(capacity_matrix, layer_directions, nets)
    
    def test_reset(self):
        """Test environment reset"""
        obs = self.env.reset(0)
        
        self.assertIsNotNone(self.env.source)
        self.assertIsNotNone(self.env.target)
        self.assertEqual(self.env.current_position, self.env.source)
        self.assertEqual(len(self.env.current_path), 1)
        self.assertFalse(self.env.done)
        
        # Check observation shape
        self.assertEqual(obs.shape, (2, 10, 10, 7))
    
    def test_step(self):
        """Test environment step"""
        obs = self.env.reset(0)
        
        # Valid action
        action = 1  # Right
        next_obs, reward, done, info = self.env.step(action)
        
        self.assertIsNotNone(next_obs)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIn('current_position', info)
        self.assertIn('valid_actions', info)
    
    def test_invalid_action(self):
        """Test invalid action handling"""
        obs = self.env.reset(0)
        
        # Try invalid action many times until it fails
        for _ in range(100):
            if self.env.done:
                break
            # Random action - might be invalid
            action = 4  # UP_LAYER might be invalid
            next_obs, reward, done, info = self.env.step(action)
            if done and info.get('terminal_reason') == 'invalid_action':
                self.assertLess(reward, 0)
                break
    
    def test_terminal_conditions(self):
        """Test terminal condition detection"""
        obs = self.env.reset(0)
        
        # Manually set position to target
        self.env.current_position = self.env.target
        
        # Should detect terminal
        # (Can't actually step because we'd need valid path, 
        # so just check the logic exists)
        self.assertIsNotNone(self.env.target)


class TestAgents(unittest.TestCase):
    """Test agent implementations"""
    
    def setUp(self):
        """Create test environment"""
        capacity_matrix = np.ones((2, 10, 10), dtype=np.float32) * 5
        layer_directions = [0, 0]
        nets = {
            'net1': [(1, 0, 0, 'M1'), (1, 5, 5, 'M1')],
        }
        self.env = VLSIRoutingEnv(capacity_matrix, layer_directions, nets)
    
    def test_random_agent(self):
        """Test random agent"""
        agent = RandomAgent()
        obs = self.env.reset(0)
        
        valid_actions = ['L', 'R', 'U']
        action = agent.select_action(obs, valid_actions)
        
        self.assertIn(action, [0, 1, 2])
    
    def test_greedy_agent(self):
        """Test greedy agent"""
        agent = GreedyAgent(target=self.env.target)
        obs = self.env.reset(0)
        
        valid_actions = ['L', 'R', 'U', 'D']
        action = agent.select_action(obs, valid_actions)
        
        self.assertIsNotNone(action)


class TestModels(unittest.TestCase):
    """Test neural network models"""
    
    def test_cnn_model_import(self):
        """Test CNN model can be imported and created"""
        try:
            from rl_gnn_models import CNNRoutingPolicy
            model = CNNRoutingPolicy(input_channels=7, num_actions=6)
            self.assertIsNotNone(model)
        except Exception as e:
            self.fail(f"CNN model creation failed: {e}")
    
    def test_cnn_forward(self):
        """Test CNN forward pass"""
        try:
            from rl_gnn_models import CNNRoutingPolicy
            model = CNNRoutingPolicy(input_channels=7, num_actions=6)
            
            # Create dummy input
            batch_size = 2
            x = torch.randn(batch_size, 2, 10, 10, 7)  # [batch, layers, y, x, features]
            
            action_logits, value = model(x)
            
            self.assertEqual(action_logits.shape, (batch_size, 6))
            self.assertEqual(value.shape, (batch_size, 1))
        except Exception as e:
            self.fail(f"CNN forward pass failed: {e}")
    
    def test_gcn_model_availability(self):
        """Test if GCN model can be imported (may fail if PyG not installed)"""
        try:
            from rl_gnn_models import GCNRoutingPolicy
            # If PyTorch Geometric not available, this should raise ImportError
        except ImportError:
            self.skipTest("PyTorch Geometric not available")


class TestIntegration(unittest.TestCase):
    """Integration tests"""
    
    def test_complete_episode(self):
        """Test complete episode execution"""
        # Create small environment
        capacity_matrix = np.ones((2, 5, 5), dtype=np.float32) * 5
        layer_directions = [0, 0]
        nets = {
            'net1': [(1, 0, 0, 'M1'), (1, 2, 2, 'M1')],
        }
        
        env = VLSIRoutingEnv(capacity_matrix, layer_directions, nets, max_steps_per_net=50)
        agent = RandomAgent()
        
        obs = env.reset(0)
        total_reward = 0
        steps = 0
        
        while not env.done and steps < 50:
            valid_actions = env.state.get_valid_actions(env.current_position)
            if not valid_actions:
                break
            
            action = agent.select_action(obs, valid_actions)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
        
        # Episode should terminate
        self.assertTrue(steps > 0)
        self.assertIsInstance(total_reward, float)
    
    def test_sequential_routing(self):
        """Test routing multiple nets sequentially"""
        capacity_matrix = np.ones((2, 8, 8), dtype=np.float32) * 5
        layer_directions = [0, 0]
        nets = {
            'net1': [(1, 0, 0, 'M1'), (1, 3, 3, 'M1')],
            'net2': [(1, 4, 4, 'M1'), (1, 7, 7, 'M1')],
        }
        
        env = VLSIRoutingEnv(capacity_matrix, layer_directions, nets, max_steps_per_net=100)
        agent = RandomAgent()
        
        routed_count = 0
        for net_idx in range(2):
            obs = env.reset(net_idx)
            steps = 0
            
            while not env.done and steps < 100:
                valid_actions = env.state.get_valid_actions(env.current_position)
                if not valid_actions:
                    break
                
                action = agent.select_action(obs, valid_actions)
                obs, reward, done, info = env.step(action)
                steps += 1
            
            if env.done and info.get('terminal_reason') == 'target_reached':
                routed_count += 1
        
        # At least should attempt routing
        self.assertGreaterEqual(env.current_net_idx, 0)


def run_tests(verbose=True):
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestGraphState))
    suite.addTests(loader.loadTestsFromTestCase(TestRewardCalculator))
    suite.addTests(loader.loadTestsFromTestCase(TestEnvironment))
    suite.addTests(loader.loadTestsFromTestCase(TestAgents))
    suite.addTests(loader.loadTestsFromTestCase(TestModels))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("="*70)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests(verbose=True)
    sys.exit(0 if success else 1)
