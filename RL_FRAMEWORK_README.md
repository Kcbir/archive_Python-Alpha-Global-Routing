# Reinforcement Learning Framework for VLSI Routing

This framework implements a complete RL solution for VLSI global routing using graph-based state representations and configurable reward systems.

## Overview

The framework addresses VLSI routing as a sequential decision-making problem where:

1. **State**: Graph/GNN representation of the routing grid (GCells) with:
   - Node features: capacity, congestion, occupancy, terminal markers
   - Edge features: routing costs, congestion, edge types (horizontal/vertical/via)
   - Dynamic updates after each net is routed

2. **Actions**: Directional movements {L, R, U, D, UP_LAYER, DOWN_LAYER}
   - Constrained by layer routing directions
   - Validated against capacity and blockages

3. **Rewards**: Multi-component formulation:
   - ✅ Large positive: Reaching target
   - ❌ Large negative: Dead ends
   - 📏 Distance-based: Manhattan distance changes
   - ⚡ Action costs: Step penalty, direction changes, vias
   - 🚦 Congestion: Penalties for high-congestion areas
   - 📊 Path quality: Efficiency bonuses

## Architecture

```
rl_graph_state.py        → Graph-based state representation
rl_reward.py             → Reward calculation system
rl_environment.py        → OpenAI Gym environment
rl_gnn_models.py         → GNN/CNN policy networks
rl_train_example.py      → Training examples and utilities
```

## Key Components

### 1. Graph State Representation (`rl_graph_state.py`)

**RoutingGraphState**: Core state representation
- **NetworkX graph**: For traditional graph algorithms
- **PyTorch Geometric**: For GNN processing
- **Feature matrices**: For CNN-based agents

```python
from rl_graph_state import RoutingGraphState

# Create state
state = RoutingGraphState(capacity_matrix, layer_directions)

# Get valid actions
actions = state.get_valid_actions(current_node)

# Apply action and move
next_node = state.apply_action(current_node, 'R')

# Update after routing
state.update_with_route(path, net_name)

# Convert to GNN format
pyg_data = state.to_pyg_hetero_data(source, target)
```

**Node Features** (11 dimensions):
```
[layer, x, y, capacity, original_capacity, congestion, 
 is_blocked, is_terminal, is_source, is_target, occupancy]
```

**Edge Features** (6 dimensions):
```
[edge_type, capacity, congestion, cost, is_occupied, occupancy_count]
```

### 2. Reward System (`rl_reward.py`)

**RewardCalculator**: Implements the specified reward formulation

```python
from rl_reward import RewardCalculator, RewardConfig

# Configure rewards
config = RewardConfig()
config.REWARD_REACH_TARGET = 1000.0
config.REWARD_DEAD_END = -1000.0
config.REWARD_CLOSER_TO_TARGET = 10.0
config.COST_VIA = -10.0

calculator = RewardCalculator(config)

# Calculate step reward
reward, breakdown = calculator.calculate_reward(
    current_node=current,
    next_node=next_pos,
    target_node=target,
    action='R',
    state=state,
    is_terminal=False
)

print(breakdown)
# {
#   'closer_to_target': 10.0,
#   'step_cost': -1.0,
#   'low_congestion': 5.0,
#   ...
# }
```

**Reward Components**:

| Component | Range | Purpose |
|-----------|-------|---------|
| Target Reached | +1000 | Terminal success |
| Dead End | -1000 | Invalid state penalty |
| Closer to Target | +10 per cell | Distance improvement |
| Detour Penalty | -15 per cell | Moving away |
| Step Cost | -1 | Action cost |
| Direction Change | -5 | Routing complexity |
| Via Cost | -10 | Layer change cost |
| Low Congestion | +5 | Encourage free paths |
| High Congestion | -20 × congestion | Avoid congested areas |
| Revisit | -50 | Prevent loops |
| Path Efficiency | ±50-100 | Final path quality |

**AdaptiveRewardCalculator**: Adjusts weights based on routing progress
- Early routing: Prioritize efficiency
- Late routing: Prioritize congestion avoidance

### 3. RL Environment (`rl_environment.py`)

**VLSIRoutingEnv**: OpenAI Gym-compatible environment

```python
from rl_environment import create_vlsi_env_from_files

# Create environment
env = create_vlsi_env_from_files(
    cap_file='data.cap',
    net_file='data.net',
    use_adaptive_rewards=True,
    max_steps_per_net=1000
)

# Standard Gym interface
obs = env.reset(net_idx=0)
action = agent.select_action(obs)
next_obs, reward, done, info = env.step(action)

# Get detailed info
print(info)
# {
#   'valid_actions': ['L', 'R', 'U'],
#   'current_position': (2, 10, 15),
#   'reward_breakdown': {...},
#   'terminal_reason': 'target_reached',
#   'path_length': 42
# }
```

**API**:
- `reset(net_idx)`: Start routing a new net
- `step(action)`: Execute action, return (obs, reward, done, info)
- `render(mode='ascii')`: Visualize current state
- `get_state_summary()`: Get routing statistics

### 4. GNN Models (`rl_gnn_models.py`)

Multiple neural network architectures for routing policies:

#### GCN-based Policy
```python
from rl_gnn_models import GCNRoutingPolicy

model = GCNRoutingPolicy(
    node_feature_dim=11,
    hidden_dim=128,
    num_layers=3,
    num_actions=6
)

# Forward pass
action_logits, value = model(graph_data, current_node_idx)
```

#### GAT-based Policy (Attention)
```python
from rl_gnn_models import GATRoutingPolicy

model = GATRoutingPolicy(
    node_feature_dim=11,
    hidden_dim=128,
    num_heads=4,
    num_actions=6
)
```

#### CNN-based Policy (Grid-based)
```python
from rl_gnn_models import CNNRoutingPolicy

model = CNNRoutingPolicy(
    input_channels=7,  # Features per cell
    num_actions=6,
    hidden_dim=128
)

# Works with feature matrix observations
action_logits, value = model(feature_matrix)
```

#### Dueling DQN
```python
from rl_gnn_models import DuelingDQN

model = DuelingDQN(
    node_feature_dim=11,
    hidden_dim=128,
    num_actions=6,
    use_gnn=True
)

q_values = model(graph_data)
```

## Usage Examples

### Example 1: Basic Environment Usage

```python
from rl_environment import create_vlsi_env_from_files, RandomAgent

# Load environment
env = create_vlsi_env_from_files('data.cap', 'data.net')

# Create agent
agent = RandomAgent()

# Route one net
obs = env.reset(net_idx=0)
done = False

while not done:
    valid_actions = env.state.get_valid_actions(env.current_position)
    action = agent.select_action(obs, valid_actions)
    obs, reward, done, info = env.step(action)
    
    if done:
        print(f"Result: {info['terminal_reason']}")
        print(f"Path length: {info.get('path_length')}")
```

### Example 2: Train DQN Agent

```python
from rl_train_example import DQNAgent, train_dqn
from rl_gnn_models import CNNRoutingPolicy

# Create model
model = CNNRoutingPolicy(input_channels=7, num_actions=6)

# Create agent
agent = DQNAgent(
    model=model,
    gamma=0.99,
    epsilon_start=1.0,
    learning_rate=0.001
)

# Train
results = train_dqn(
    env=env,
    agent=agent,
    num_episodes=1000,
    batch_size=32
)

# Evaluate
from rl_train_example import evaluate_agent
eval_results = evaluate_agent(env, agent, num_episodes=10)
print(f"Success rate: {eval_results['success_rate']:.2%}")
```

### Example 3: Custom Reward Configuration

```python
from rl_reward import RewardConfig
from rl_environment import create_vlsi_env_from_files

# Customize rewards
config = RewardConfig()
config.REWARD_REACH_TARGET = 2000.0  # More emphasis on success
config.PENALTY_HIGH_CONGESTION = -50.0  # Strongly avoid congestion
config.COST_VIA = -20.0  # Discourage layer changes

env = create_vlsi_env_from_files(
    'data.cap', 'data.net',
    reward_config=config,
    use_adaptive_rewards=True
)
```

### Example 4: Graph Visualization

```python
import matplotlib.pyplot as plt
import networkx as nx

# Get NetworkX graph
graph = env.state.nx_graph

# Visualize layer
layer = 1
layer_nodes = [(l, y, x) for l, y, x in graph.nodes() if l == layer]
subgraph = graph.subgraph(layer_nodes)

pos = {(l, y, x): (x, y) for l, y, x in layer_nodes}
nx.draw(subgraph, pos, node_size=20, with_labels=False)
plt.title(f"Layer {layer} Routing Graph")
plt.show()
```

## Training Workflow

### Complete Training Pipeline

```python
# 1. Load data
env = create_vlsi_env_from_files('data.cap', 'data.net')

# 2. Create model (choose one)
model = CNNRoutingPolicy(...)  # OR
model = GCNRoutingPolicy(...)  # OR
model = GATRoutingPolicy(...)

# 3. Create RL agent (choose algorithm)
from stable_baselines3 import PPO  # Example with SB3
agent = PPO('MlpPolicy', env, verbose=1)

# 4. Train
agent.learn(total_timesteps=100000)

# 5. Evaluate
obs = env.reset()
for _ in range(1000):
    action, _ = agent.predict(obs)
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()

# 6. Save model
agent.save("vlsi_routing_agent")
```

## State Update Process

After each net is routed:

```python
# 1. Route net (accumulate path)
path = [source]
current = source

while current != target:
    action = select_action(state, current)
    current = state.apply_action(current, action)
    path.append(current)

# 2. Update state
state.update_with_route(path, net_name)

# This updates:
# - Capacity matrix (decrease by 1 for used cells)
# - Occupancy matrix (increment for used cells)
# - Congestion values (recalculated)
# - Edge costs (increased for congested edges)
# - Graph structure (edge weights updated)

# 3. Next net sees updated state
next_net_obs = env.reset(net_idx + 1)
# This net must now route around previously routed nets
```

## Integration with Existing Router

To integrate with your existing `router.py`:

```python
# In router.py, add RL-based routing function

from rl_environment import VLSIRoutingEnv
from rl_gnn_models import GCNRoutingPolicy

def route_net_with_rl(net_name, net_data, capacity_matrix, layer_directions, agent):
    """Route a single net using RL agent"""
    
    # Create temporary environment for this net
    env = VLSIRoutingEnv(
        capacity_matrix=capacity_matrix,
        layer_directions=layer_directions,
        nets={net_name: net_data},
        max_steps_per_net=1000
    )
    
    obs = env.reset(0)
    path = [env.source]
    done = False
    
    while not done:
        valid_actions = env.state.get_valid_actions(env.current_position)
        action = agent.select_action(obs, valid_actions)
        obs, reward, done, info = env.step(action)
        
        if done and info['terminal_reason'] == 'target_reached':
            return env.current_path, True
    
    return [], False


# Use in main routing loop
def route_circuit_rl(cap_file, net_file, agent):
    cap_data = read_cap(cap_file)
    net_data = read_net(net_file)
    
    capacity_matrix = cap_data['cap'].copy()
    
    for net_name, terminals in net_data.items():
        path, success = route_net_with_rl(
            net_name, terminals, capacity_matrix,
            cap_data['layerDirections'], agent
        )
        
        if success:
            # Update capacity
            for node in path:
                layer, y, x = node
                capacity_matrix[layer, y, x] -= 1
            
            print(f"✓ Routed {net_name}")
        else:
            print(f"✗ Failed {net_name}")
```

## Requirements

```bash
# Core dependencies
pip install numpy torch networkx gym

# For GNN models (optional but recommended)
pip install torch-geometric

# For visualization
pip install matplotlib

# For advanced RL algorithms
pip install stable-baselines3
```

## Performance Tips

1. **Start with CNN models**: Faster than GNN for initial experiments
2. **Use adaptive rewards**: Better for sequential routing
3. **Batch training**: Process multiple nets in parallel when possible
4. **Transfer learning**: Pre-train on simpler circuits
5. **Curriculum learning**: Start with 2-pin nets, progress to multi-pin

## Advanced Features

### Custom GNN Architecture

```python
import torch.nn as nn
from torch_geometric.nn import MessagePassing

class CustomRoutingGNN(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')
        self.lin = nn.Linear(in_channels, out_channels)
        
    def forward(self, x, edge_index, edge_attr):
        # Custom message passing
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_j, edge_attr):
        # Combine node and edge features
        return self.lin(x_j) * edge_attr
```

### Multi-Agent Routing

```python
# Route multiple nets simultaneously
class MultiAgentEnv(VLSIRoutingEnv):
    def __init__(self, num_agents=4, **kwargs):
        super().__init__(**kwargs)
        self.num_agents = num_agents
        self.active_nets = []
    
    def step_all(self, actions):
        """Step all agents simultaneously"""
        observations = []
        rewards = []
        dones = []
        infos = []
        
        for i, action in enumerate(actions):
            obs, r, d, info = self.step_agent(i, action)
            observations.append(obs)
            rewards.append(r)
            dones.append(d)
            infos.append(info)
        
        return observations, rewards, dones, infos
```

## Future Enhancements

- [ ] Hierarchical RL for multi-pin nets
- [ ] Graph attention with edge features
- [ ] Curriculum learning framework
- [ ]Transfer learning across circuits
- [ ] Multi-agent coordination
- [ ] Real-time constraint satisfaction
- [ ] Integration with physical design rules

## Citation

If you use this framework, please cite:

```bibtex
@software{vlsi_routing_rl,
  title={Reinforcement Learning Framework for VLSI Global Routing},
  author={Your Name},
  year={2026},
  url={https://github.com/your-repo}
}
```

## License

[Your License Here]

## Contact

For questions and issues, please open an issue on GitHub or contact [your email].
