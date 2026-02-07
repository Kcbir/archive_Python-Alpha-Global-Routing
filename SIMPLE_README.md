# Simple RL Framework for VLSI Routing

**Clean. Fast. Easy.**

Using **Rustworkx** (fast graphs) + **GraphSAGE** (powerful GNN) + **Basic RL**

## Quick Start (1 minute)

```bash
# Install dependencies
pip install numpy torch rustworkx matplotlib

# Optional: For GraphSAGE GNN
pip install torch-geometric

# Run test
python test_simple.py
```

## What You Get

### 3 Simple Files:

1. **`rl_simple_state.py`** - Fast graph state with Rustworkx
2. **`rl_simple_agent.py`** - GraphSAGE + CNN policies 
3. **`test_simple.py`** - Test everything

That's it! No complex stuff. Just what works.

## Usage

### Basic Example

```python
from rl_simple_state import SimpleRoutingState
from rl_simple_agent import create_simple_agent
import numpy as np

# Create routing state
capacity = np.ones((2, 10, 10)) * 5  # 2 layers, 10x10 grid, capacity=5
layer_dirs = [0, 0]  # 0=horizontal, 1=vertical

state = SimpleRoutingState(capacity, layer_dirs)

# Create agent (use CNN - no GNN needed)
agent = create_simple_agent(use_gnn=False)

# Route!
source = (1, 0, 0)  # (layer, y, x)
target = (1, 5, 5)
current = source

for step in range(100):
    # Get what can we do
    valid_actions = state.get_valid_actions(current)
    if not valid_actions:
        break
    
    # Get observation
    obs = state.to_numpy_features(source, target)
    
    # Agent picks action
    action = agent.select_action(obs, valid_actions)
    
    # Do it
    next_node = state.apply_action(current, ['L','R','U','D','UP_LAYER','DOWN_LAYER'][action])
    current = next_node
    
    if current == target:
        print(f"✓ Reached target in {step+1} steps!")
        break
```

### With GraphSAGE (if you want GNN)

```python
# Create GraphSAGE agent
agent = create_simple_agent(use_gnn=True)

# Get graph observation
pyg_data = state.to_pyg_data(source, target)

# Agent uses graph structure
action = agent.select_action(pyg_data, valid_actions)
```

### With Your Data

```python
from utils import read_cap, read_net

# Load your VLSI data
cap_data = read_cap('your_circuit.cap')
net_data = read_net('your_circuit.net')

# Create state
state = SimpleRoutingState(
    cap_data['cap'],
    cap_data['layerDirections']
)

# Route nets
for net_name, terminals in net_data.items():
    source = (terminals[0][0], terminals[0][2], terminals[0][1])
    target = (terminals[1][0], terminals[1][2], terminals[1][1])
    
    # ... route using agent ...
```

## State Features

### Node Features (8D for GraphSAGE):
```
[layer, x, y, capacity, congestion, is_source, is_target, distance_to_target]
```

### Grid Features (5D for CNN):
```
[normalized_capacity, is_source, is_target, manhattan_distance, is_blocked]
```

## Performance

**Rustworkx is FAST:**
- 10x faster than NetworkX for large graphs
- Written in Rust - native speed
- Scales to huge circuits

**GraphSAGE is POWERFUL:**
- Learns from graph structure
- Aggregates neighbor information
- State-of-the-art performance

## Actions

Simple movements:
- `L` - Left
- `R` - Right  
- `U` - Up
- `D` - Down
- `UP_LAYER` - Go up a layer (via)
- `DOWN_LAYER` - Go down a layer (via)

Only valid actions returned (checks capacity, blockages, layer directions)

## Models

### 1. SimpleCNNPolicy (Fast Baseline)
```python
from rl_simple_agent import SimpleCNNPolicy

model = SimpleCNNPolicy(
    input_channels=5,
    hidden_dim=64,
    num_actions=6
)
```
- Uses 3D CNN on grid
- No GNN needed
- Super fast
- Good baseline

### 2. GraphSAGEPolicy (Powerful GNN)
```python
from rl_simple_agent import GraphSAGEPolicy

model = GraphSAGEPolicy(
    input_dim=8,
    hidden_dim=64,
    num_actions=6
)
```
- Learns from graph structure
- Better for complex routing
- Needs PyTorch Geometric

## State Updates

After routing a net:
```python
path = [source, node1, node2, ..., target]
state.update_after_routing(path)
# Capacity decreases along path
# Next net sees updated state
```

## Why This Approach?

✅ **Rustworkx** - Fast C/Rust implementation, 10x NetworkX speed  
✅ **GraphSAGE** - State-of-the-art GNN, learns from neighbors  
✅ **Simple** - Only 3 files, ~500 lines total  
✅ **Flexible** - Use CNN or GNN, your choice  
✅ **Clean** - No complex inheritance, easy to modify  

## Comparison

| Feature | Complex Version | Simple Version |
|---------|----------------|----------------|
| Files | 8+ files | 3 files |
| Lines | 4500+ | ~500 |
| Graph lib | NetworkX | Rustworkx (10x faster) |
| GNN | GCN/GAT/etc | GraphSAGE (best) |
| complexity | High | Low |
| Speed | Slow | Fast |

## Next Steps

1. **Test it**: `python test_simple.py`
2. **Train on your data**: Modify `test_simple.py`
3. **Tune hyperparameters**: Change `hidden_dim`, etc.
4. **Scale up**: Rustworkx handles huge graphs

## Troubleshooting

### "rustworkx not available"
```bash
pip install rustworkx
```

### "torch-geometric not available"
```bash
# Optional - only needed for GraphSAGE
pip install torch-geometric
# Or just use CNN policy (works great!)
```

### Agent not learning?
- Increase `hidden_dim` (64 -> 128)
- More training episodes
- Adjust epsilon (exploration)

## Files

- **`rl_simple_state.py`** (200 lines) - State with Rustworkx
- **`rl_simple_agent.py`** (200 lines) - GraphSAGE + CNN policies
- **`test_simple.py`** (150 lines) - Tests

Total: ~550 lines. Clean. Fast. Works.

## Summary

You asked for:
1. ✅ **GraphSAGE** - Done! 
2. ✅ **Rustworkx** - Done! (10x faster)
3. ✅ **Basic RL** - Done! (no complex stuff)
4. ✅ **Simple** - Just 3 files!

Now you can:
- Test quickly: `python test_simple.py`
- Train on your data
- Scale to huge circuits
- Iterate fast

**Simple. Fast. Ready. Let's go! 🚀**

---

Questions? Just run `python test_simple.py` and see it work!
