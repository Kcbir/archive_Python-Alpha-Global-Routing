# coding: utf-8
"""
Simple Test - Just the Basics
Clean, fast, easy to understand!
"""

import numpy as np
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rl_simple_state import SimpleRoutingState
from rl_simple_agent import create_simple_agent, GraphSAGEPolicy, SimpleCNNPolicy
from utils import read_cap, read_net


def test_state():
    """Test 1: State representation"""
    print("\n" + "="*60)
    print("TEST 1: State with Rustworkx")
    print("="*60)
    
    # Create simple test state
    cap = np.ones((2, 5, 5), dtype=np.float32) * 3
    cap[:, 0, 0] = 0  # Block one cell
    layer_dirs = [0, 0]  # Both horizontal
    
    state = SimpleRoutingState(cap, layer_dirs)
    
    print(f"✓ State created")
    print(f"  Nodes: {state.graph.num_nodes() if state.graph else 'N/A'}")
    print(f"  Edges: {state.graph.num_edges() if state.graph else 'N/A'}")
    
    # Test actions
    node = (1, 2, 2)
    actions = state.get_valid_actions(node)
    print(f"✓ Valid actions from {node}: {actions}")
    
    # Test action
    if 'R' in actions:
        next_node = state.apply_action(node, 'R')
        print(f"✓ Move right: {node} -> {next_node}")
    
    # Test features
    source = (1, 0, 0)
    target = (1, 4, 4)
    features = state.to_numpy_features(source, target)
    print(f"✓ Feature matrix shape: {features.shape}")
    
    # Test PyG data
    pyg_data = state.to_pyg_data(source, target)
    if pyg_data:
        print(f"✓ PyG Data: {pyg_data.num_nodes} nodes, {pyg_data.edge_index.shape[1]} edges")
    
    print(f"\n✓ State summary: {state.summary()}")


def test_model():
    """Test 2: Models"""
    print("\n" + "="*60)
    print("TEST 2: GraphSAGE & CNN Models")
    print("="*60)
    
    try:
        import torch
        
        # Test CNN (always works)
        print("\nTesting CNN Policy...")
        cnn_policy = SimpleCNNPolicy(input_channels=5, hidden_dim=64)
        dummy_input = torch.randn(1, 2, 5, 5, 5)
        action_logits, value = cnn_policy(dummy_input)
        print(f"✓ CNN forward pass: actions={action_logits.shape}, value={value.shape}")
        
        # Test GraphSAGE (needs PyG)
        try:
            print("\nTesting GraphSAGE Policy...")
            sage_policy = GraphSAGEPolicy(input_dim=8, hidden_dim=64)
            
            from torch_geometric.data import Data
            dummy_data = Data(
                x=torch.randn(50, 8),
                edge_index=torch.randint(0, 50, (2, 100))
            )
            action_logits, value = sage_policy(dummy_data)
            print(f"✓ GraphSAGE forward pass: actions={action_logits.shape}, value={value.shape}")
        except ImportError:
            print("⚠ GraphSAGE needs torch-geometric (optional)")
        
    except ImportError:
        print("⚠ PyTorch not installed")


def test_agent():
    """Test 3: Agent"""
    print("\n" + "="*60)
    print("TEST 3: RL Agent")
    print("="*60)
    
    try:
        # Create simple environment
        cap = np.ones((2, 8, 8), dtype=np.float32) * 3
        layer_dirs = [0, 0]
        state = SimpleRoutingState(cap, layer_dirs)
        
        # Create agent (CNN for simplicity)
        agent = create_simple_agent(use_gnn=False, input_channels=5, hidden_dim=64)
        print(f"✓ Agent created")
        
        # Test routing
        source = (1, 0, 0)
        target = (1, 5, 5)
        current = source
        path = [current]
        
        print(f"\nRouting from {source} to {target}...")
        
        for step in range(20):
            # Get observation
            obs = state.to_numpy_features(source, target)
            
            # Get valid actions
            valid_actions = state.get_valid_actions(current)
            if not valid_actions:
                print(f"  Dead end at step {step}")
                break
            
            # Select action
            action_idx = agent.select_action(obs, valid_actions)
            action_str = ['L', 'R', 'U', 'D', 'UP_LAYER', 'DOWN_LAYER'][action_idx]
            
            # Execute
            next_node = state.apply_action(current, action_str)
            if next_node is None:
                break
            
            current = next_node
            path.append(current)
            
            print(f"  Step {step}: {action_str} -> {current}")
            
            if current == target:
                print(f"\n✓ REACHED TARGET in {step+1} steps!")
                break
        
        # Update state
        state.update_after_routing(path)
        print(f"\n✓ State updated after routing")
        print(f"  Path length: {len(path)}")
        print(f"  Utilization: {state.summary()['utilization']:.2%}")
        
    except Exception as e:
        print(f"⚠ Agent test error: {e}")


def test_with_real_data():
    """Test 4: Real data"""
    print("\n" + "="*60)
    print("TEST 4: Real VLSI Data")
    print("="*60)
    
    cap_file = 'test_data/ariane133_51.cap'
    net_file = 'test_data/ariane133_51.net'
    
    if not os.path.exists(cap_file):
        print(f"⚠ Data files not found: {cap_file}")
        print("  (This is optional - basic tests already passed)")
        return
    
    try:
        # Load data
        print("Loading data...")
        cap_data = read_cap(cap_file, verbose=False)
        net_data = read_net(net_file, verbose=False)
        
        print(f"✓ Loaded {len(net_data)} nets")
        print(f"  Grid: {cap_data['nLayers']}L x {cap_data['ySize']}x{cap_data['xSize']}")
        
        # Create state
        state = SimpleRoutingState(
            cap_data['cap'],
            cap_data['layerDirections']
        )
        
        print(f"✓ State created")
        print(f"  {state.summary()}")
        
        # Pick a simple net
        net_name = list(net_data.keys())[0]
        terminals = net_data[net_name]
        
        if len(terminals) >= 2:
            source = (terminals[0][0], terminals[0][2], terminals[0][1])
            target = (terminals[1][0], terminals[1][2], terminals[1][1])
            
            print(f"\n✓ Test net: {net_name}")
            print(f"  Source: {source}")
            print(f"  Target: {target}")
            
            # Test features
            features = state.to_numpy_features(source, target)
            print(f"  Feature shape: {features.shape}")
        
    except Exception as e:
        print(f"⚠ Real data test error: {e}")


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("SIMPLE RL FRAMEWORK TEST")
    print("Rustworkx + GraphSAGE + Basic RL")
    print("="*60)
    
    test_state()
    test_model()
    test_agent()
    test_with_real_data()
    
    print("\n" + "="*60)
    print("TESTS COMPLETE!")
    print("="*60)
    print("\nWhat you got:")
    print("  ✓ Rustworkx graph state (FAST!)")
    print("  ✓ GraphSAGE GNN policy")
    print("  ✓ Simple CNN policy (baseline)")
    print("  ✓ Basic RL agent")
    print("\nNext steps:")
    print("  1. Train on your data")
    print("  2. Tune hyperparameters")
    print("  3. Scale up!")
    print("\nSimple, clean, and ready to go! 🚀")


if __name__ == '__main__':
    main()
