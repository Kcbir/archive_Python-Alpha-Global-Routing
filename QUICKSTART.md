# Quick Start Guide: RL Framework for VLSI Routing

This guide will get you started with the RL framework in 5 minutes.

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Optional: Install PyTorch Geometric for GNN models
pip install torch-geometric torch-scatter torch-sparse

# Optional: Install Stable Baselines 3 for advanced RL
pip install stable-baselines3
```

## Test the Framework

Run the test suite to ensure everything is working:

```bash
python rl_test_suite.py
```

Expected output: All tests should pass (some may be skipped if PyTorch Geometric is not installed).

## Example 1: Basic Usage (2 minutes)

```python
from rl_environment import create_vlsi_env_from_files, RandomAgent

# Load your data
env = create_vlsi_env_from_files(
    cap_file='test_data/ariane133_51.cap',
    net_file='test_data/ariane133_51.net'
)

# Create a simple agent
agent = RandomAgent()

# Route one net
obs = env.reset(net_idx=0)
print(f"Routing net: {env.current_net_name}")
print(f"From {env.source} to {env.target}")

for step in range(100):
    # Get valid moves
    valid_actions = env.state.get_valid_actions(env.current_position)
    
    if not valid_actions:
        print("Dead end!")
        break
    
    # Select and execute action
    action = agent.select_action(obs, valid_actions)
    obs, reward, done, info = env.step(action)
    
    print(f"Step {step}: {env.action_to_str[action]} -> {env.current_position}, Reward: {reward:.1f}")
    
    if done:
        print(f"\n✓ Finished: {info['terminal_reason']}")
        if info.get('path_length'):
            print(f"Path length: {info['path_length']}")
        break
```

## Example 2: Train a DQN Agent (5 minutes)

```python
from rl_train_example import example_train_dqn

# This will:
# 1. Create environment
# 2. Create CNN-based DQN agent
# 3. Train for 100 episodes
# 4. Evaluate performance

agent, results = example_train_dqn()

# View training progress
import matplotlib.pyplot as plt
plt.plot(results['episode_rewards'])
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Training Progress')
plt.show()
```

## Example 3: Visualize Routing (3 minutes)

```python
from rl_visualization import visualize_episode
from rl_environment import create_vlsi_env_from_files, RandomAgent

# Create environment
env = create_vlsi_env_from_files(
    'test_data/ariane133_51.cap',
    'test_data/ariane133_51.net'
)

# Visualize routing process
agent = RandomAgent()
visualize_episode(env, agent, save_dir='visualizations')

# This creates PNG files showing:
# - Initial state
# - Progress every 10 steps
# - Final 3D path
# - State summary
```

## Example 4: Custom Rewards (2 minutes)

```python
from rl_reward import RewardConfig
from rl_environment import create_vlsi_env_from_files

# Customize reward structure
config = RewardConfig()
config.REWARD_REACH_TARGET = 2000.0        # Emphasize success
config.PENALTY_HIGH_CONGESTION = -50.0     # Avoid congestion strongly
config.COST_VIA = -20.0                    # Discourage layer changes
config.REWARD_CLOSER_TO_TARGET = 15.0      # Reward efficient paths

# Create environment with custom rewards
env = create_vlsi_env_from_files(
    'test_data/ariane133_51.cap',
    'test_data/ariane133_51.net',
    reward_config=config,
    use_adaptive_rewards=True
)

# Now env.step() will use your custom rewards
```

## Example 5: Graph State Inspection (2 minutes)

```python
from rl_graph_state import RoutingGraphState
from utils import read_cap

# Load capacity data
cap_data = read_cap('test_data/ariane133_51.cap')

# Create graph state
state = RoutingGraphState(
    capacity_matrix=cap_data['cap'],
    layer_directions=cap_data['layerDirections']
)

# Inspect graph
print(f"Nodes: {state.nx_graph.number_of_nodes()}")
print(f"Edges: {state.nx_graph.number_of_edges()}")

# Get state summary
summary = state.get_state_summary()
print(f"\nCapacity utilization: {summary['utilization']:.2%}")
print(f"Average congestion: {summary['avg_congestion']:.2%}")
print(f"Blocked cells: {summary['blocked_cells']}")

# Visualize a layer
print(state.visualize_layer(layer=1))
```

## Example 6: Using with GNN Models (Advanced)

```python
from rl_gnn_models import GCNRoutingPolicy, create_model
import torch

# Create GCN-based policy
model = GCNRoutingPolicy(
    node_feature_dim=11,    # From GCellNode features
    hidden_dim=128,
    num_layers=3,
    num_actions=6
)

# Or use factory function
model = create_model('gcn', hidden_dim=128, num_layers=3)

# Get graph data from state
graph_data = env.get_graph_observation()

if graph_data is not None:
    # Forward pass
    action_logits, value = model(graph_data)
    
    # Select action
    action_probs = torch.softmax(action_logits, dim=-1)
    action = torch.argmax(action_probs).item()
    
    # Use in environment
    obs, reward, done, info = env.step(action)
```

## Common Tasks

### Task 1: Route All Nets with an Agent

```python
from rl_environment import VLSIRoutingEnv
from rl_train_example import GreedyAgent

env = create_vlsi_env_from_files('data.cap', 'data.net')
agent = GreedyAgent(target=None)  # Will update target for each net

for net_idx in range(len(env.net_names)):
    obs = env.reset(net_idx)
    agent.target = env.target  # Update target
    
    done = False
    while not done:
        valid_actions = env.state.get_valid_actions(env.current_position)
        if not valid_actions:
            break
        
        action = agent.select_action(obs, valid_actions)
        obs, reward, done, info = env.step(action)
    
    if done and info.get('terminal_reason') == 'target_reached':
        print(f"✓ Routed {env.current_net_name}")
    else:
        print(f"✗ Failed {env.current_net_name}")
```

### Task 2: Analyze Reward Breakdown

```python
from rl_reward import RewardCalculator

calculator = RewardCalculator()

# During routing
reward, breakdown = calculator.calculate_reward(
    current_node=(1, 5, 5),
    next_node=(1, 5, 6),
    target_node=(1, 10, 10),
    action='R',
    state=env.state,
    is_terminal=False
)

# Print detailed breakdown
print("Reward Breakdown:")
for component, value in breakdown.items():
    print(f"  {component}: {value:.2f}")
print(f"Total: {reward:.2f}")
```

### Task 3: Compare Different Agents

```python
from rl_train_example import example_compare_agents

# This will test Random vs Greedy agents
results = example_compare_agents()

# Results dict contains success rates and rewards for each agent
for agent_name, metrics in results.items():
    print(f"\n{agent_name} Agent:")
    print(f"  Success Rate: {metrics['success_rate']:.2%}")
    print(f"  Avg Reward: {metrics['avg_reward']:.2f}")
    print(f"  Avg Path Length: {metrics['avg_path_length']:.1f}")
```

### Task 4: Visualize Training Progress

```python
from rl_visualization import RoutingVisualizer
import matplotlib.pyplot as plt

# After training
visualizer = RoutingVisualizer(env.state)

# Plot metrics
fig = visualizer.plot_training_metrics(
    episode_rewards=results['episode_rewards'],
    episode_lengths=results['episode_lengths'],
    losses=results['losses'],
    window=10
)
plt.savefig('training_progress.png', dpi=300)
```

### Task 5: Create Custom Agent

```python
class MyCustomAgent:
    """Your custom routing agent"""
    
    def __init__(self):
        # Initialize your agent
        pass
    
    def select_action(self, observation, valid_actions):
        """
        Select action based on observation and valid actions
        
        Args:
            observation: State feature matrix [layers, y, x, features]
            valid_actions: List of valid action strings ['L', 'R', 'U', ...]
            
        Returns:
            action: Integer action index (0-5)
        """
        # Map actions
        action_map = {
            'L': 0, 'R': 1, 'U': 2, 'D': 3,
            'UP_LAYER': 4, 'DOWN_LAYER': 5
        }
        
        # Your logic here
        # For example: analyze observation, compute best action
        
        # Return action index
        return action_map[valid_actions[0]]

# Use it
agent = MyCustomAgent()
env.reset(0)
action = agent.select_action(obs, ['L', 'R'])
```

## Integration with Existing Router

To use RL in your existing routing pipeline:

```python
# In your router.py or main routing script
from rl_environment import VLSIRoutingEnv
from rl_gnn_models import create_model

# Load your trained agent
model = create_model('cnn', input_channels=7, num_actions=6)
model.load_state_dict(torch.load('trained_agent.pth'))

# Create agent wrapper
class RLRoutingAgent:
    def __init__(self, model):
        self.model = model
        self.model.eval()
    
    def route_net(self, env, net_idx):
        """Route one net using RL policy"""
        obs = env.reset(net_idx)
        path = [env.source]
        
        with torch.no_grad():
            while not env.done:
                valid_actions = env.state.get_valid_actions(env.current_position)
                if not valid_actions:
                    return None, False
                
                # Get action from model
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                action_logits, _ = self.model(obs_tensor)
                action_probs = torch.softmax(action_logits, dim=-1)[0]
                
                # Mask invalid actions
                valid_indices = [action_map[a] for a in valid_actions]
                masked_probs = torch.zeros_like(action_probs)
                masked_probs[valid_indices] = action_probs[valid_indices]
                
                # Select action
                action = torch.argmax(masked_probs).item()
                
                obs, reward, done, info = env.step(action)
                path.append(env.current_position)
                
                if done:
                    success = (info['terminal_reason'] == 'target_reached')
                    return path, success
        
        return None, False

# Use in your routing
agent = RLRoutingAgent(model)
for net_idx in range(len(env.net_names)):
    path, success = agent.route_net(env, net_idx)
    if success:
        # Update your capacity matrix, save path, etc.
        pass
```

## Troubleshooting

### Issue: "PyTorch Geometric not available"

**Solution**: This is optional. Use CNN-based models instead of GNN models.

```python
# Use this
from rl_gnn_models import CNNRoutingPolicy
model = CNNRoutingPolicy(input_channels=7, num_actions=6)

# Instead of
# from rl_gnn_models import GCNRoutingPolicy
```

### Issue: "No valid actions - dead end"

**Cause**: Agent hit blockage or ran out of capacity.

**Solution**: 
1. Increase `max_steps_per_net` in environment
2. Improve agent policy
3. Check if nets are routably placed

### Issue: Training is slow

**Solutions**:
1. Reduce `capacity_matrix` resolution
2. Use smaller `hidden_dim` in models
3. Reduce `num_layers` in GNN
4. Use GPU: `model.to('cuda')`

### Issue: Agent always fails

**Solutions**:
1. Increase exploration: Higher `epsilon_start`
2. Adjust rewards: Increase `REWARD_CLOSER_TO_TARGET`
3. Train longer: More episodes
4. Check if problem is feasible

## Next Steps

1. **Read Full Documentation**: See [RL_FRAMEWORK_README.md](RL_FRAMEWORK_README.md)
2. **Run Examples**: Execute `python rl_train_example.py --example 1`
3. **Experiment**: Modify reward configuration
4. **Train**: Use your own circuit data
5. **Optimize**: Try different model architectures

## Key Files Reference

- `rl_graph_state.py` - State representation
- `rl_reward.py` - Reward calculation
- `rl_environment.py` - RL environment
- `rl_gnn_models.py` - Neural network models
- `rl_train_example.py` - Training examples
- `rl_visualization.py` - Visualization tools
- `rl_test_suite.py` - Unit tests

## Getting Help

1. Run tests: `python rl_test_suite.py`
2. Check documentation: `RL_FRAMEWORK_README.md`
3. Try examples: `python rl_train_example.py`
4. Review visualizations: Use `rl_visualization.py`

## Performance Tips

1. **Start Simple**: Use RandomAgent or GreedyAgent first
2. **Use CNN**: Faster than GNN for initial experiments
3. **Tune Rewards**: Adjust RewardConfig for your use case
4. **Visualize**: Use visualizations to debug
5. **Test Incrementally**: Route simple nets first

---

**Ready to start?** Run this now:

```bash
python rl_train_example.py --example 1
```

This will demonstrate basic environment usage with your data!
