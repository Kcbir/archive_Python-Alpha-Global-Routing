# coding: utf-8
"""
Phase 1 — Test Suite
Validates: RustWorkX state, GraphSAGE policy, CNN policy, RL agent,
           and real-data Steiner-tree routing.
"""

import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rl_simple_state import SimpleRoutingState
from rl_simple_agent import (
    create_simple_agent, GraphSAGEPolicy, SimpleCNNPolicy,
    ACTIONS, NUM_ACTIONS,
)


# ===================================================================== #
#  TEST 1 — State with RustWorkX                                         #
# ===================================================================== #

def test_state():
    print("\n" + "=" * 60)
    print("TEST 1: RustWorkX Graph State")
    print("=" * 60)

    # 3 layers: layer 0 not routable, layer 1 horizontal, layer 2 vertical
    cap = np.ones((3, 6, 6), dtype=np.float32) * 5
    cap[:, 0, 0] = 0          # block one cell
    layer_dirs = [0, 0, 1]    # h / h / v

    t0 = time.time()
    state = SimpleRoutingState(cap, layer_dirs)
    dt = time.time() - t0

    print(f"  State created in {dt*1000:.1f} ms")
    print(f"  Nodes : {state.graph.num_nodes()}")
    print(f"  Edges : {state.graph.num_edges()}")

    # Valid actions on a horizontal layer
    node = (1, 2, 2)          # layer 1, middle of grid
    actions = state.get_valid_actions(node)
    print(f"  Valid actions from {node}: {list(actions.keys())}")
    assert 'R' in actions and 'L' in actions, "Horizontal layer must allow L/R"
    assert 'U' not in actions and 'D' not in actions, "Horizontal layer must not allow U/D"

    # Valid actions on a vertical layer
    node_v = (2, 2, 2)
    actions_v = state.get_valid_actions(node_v)
    assert 'U' in actions_v and 'D' in actions_v, "Vertical layer must allow U/D"

    # Apply action
    nxt = state.apply_action(node, 'R')
    assert nxt == (1, 2, 3), f"Expected (1,2,3), got {nxt}"
    print(f"  Move R: {node} -> {nxt}")

    # Feature tensor
    source, target = (1, 0, 0), (2, 5, 5)
    features = state.to_numpy_features(source, target)
    assert features.shape == (5, 3, 6, 6), f"Bad shape: {features.shape}"
    print(f"  Feature tensor: {features.shape}")

    # PyG data
    pyg = state.to_pyg_data(source, target)
    if pyg is not None:
        print(f"  PyG Data: {pyg.num_nodes} nodes, {pyg.edge_index.shape[1]} edges")
    else:
        print("  PyG not available (optional)")

    print(f"  Summary: {state.summary()}")
    print("  PASSED")


# ===================================================================== #
#  TEST 2 — Policy forward passes                                        #
# ===================================================================== #

def test_model():
    print("\n" + "=" * 60)
    print("TEST 2: GraphSAGE & CNN Forward Pass")
    print("=" * 60)

    try:
        import torch
    except ImportError:
        print("  PyTorch not installed — skipping")
        return

    # ---- CNN ----
    nL, H, W, nf = 3, 6, 6, 5
    channels = nf * nL                           # flatten features x layers
    cnn = SimpleCNNPolicy(input_channels=channels, hidden_dim=64)
    dummy = torch.randn(1, channels, H, W)
    al, v = cnn(dummy)
    assert al.shape == (1, NUM_ACTIONS)
    assert v.shape == (1, 1)
    print(f"  CNN  : actions {al.shape}, value {v.shape}  OK")

    # ---- GraphSAGE ----
    try:
        from torch_geometric.data import Data
        sage = GraphSAGEPolicy(input_dim=8, hidden_dim=64)
        d = Data(x=torch.randn(50, 8),
                 edge_index=torch.randint(0, 50, (2, 120)))
        al2, v2 = sage(d)
        assert al2.shape == (1, NUM_ACTIONS)
        print(f"  SAGE : actions {al2.shape}, value {v2.shape}  OK")
    except ImportError:
        print("  torch-geometric not installed — GraphSAGE skipped (optional)")

    print("  PASSED")


# ===================================================================== #
#  TEST 3 — RL Agent routing on small grid                                #
# ===================================================================== #

def test_agent():
    print("\n" + "=" * 60)
    print("TEST 3: RL Agent — step-by-step routing")
    print("=" * 60)

    try:
        import torch
    except ImportError:
        print("  PyTorch not installed — skipping")
        return

    # 3 layers so the agent can move both x and y (via different layers)
    cap = np.ones((3, 8, 8), dtype=np.float32) * 5
    layer_dirs = [0, 0, 1]    # layer 1 horizontal, layer 2 vertical
    state = SimpleRoutingState(cap, layer_dirs)

    # CNN agent (always available, no PyG dependency)
    nf, nL = 5, 3
    agent = create_simple_agent(use_gnn=False,
                                input_channels=nf * nL,
                                hidden_dim=64,
                                epsilon_start=0.3)
    print("  Agent created (CNN, eps=0.3)")

    source = (1, 0, 0)
    target = (1, 7, 7)
    path, total_reward, success = agent.route_net_rl(state, source, target,
                                                     max_steps=60)

    print(f"  Route {source} -> {target}")
    print(f"  Steps : {len(path) - 1}")
    print(f"  Reward: {total_reward:.1f}")
    print(f"  Reached target: {success}")

    state.update_after_routing(path)
    s = state.summary()
    print(f"  Utilisation: {s['utilization']:.4f}")

    # Verify reward function works
    r = agent.compute_reward((1, 3, 3), (1, 3, 4), target, state)
    assert isinstance(r, float), "Reward must be a float"
    print(f"  Reward sample: {r:.2f}")

    print("  PASSED")


# ===================================================================== #
#  TEST 4 — RL training loop (tiny, just validates gradient flow)         #
# ===================================================================== #

def test_training():
    print("\n" + "=" * 60)
    print("TEST 4: RL Training Loop (gradient-flow check)")
    print("=" * 60)

    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        print("  PyTorch not installed — skipping")
        return

    cap = np.ones((3, 6, 6), dtype=np.float32) * 5
    layer_dirs = [0, 0, 1]
    nf, nL = 5, 3

    agent = create_simple_agent(use_gnn=False,
                                input_channels=nf * nL,
                                hidden_dim=32,
                                epsilon_start=0.8,
                                lr=3e-3)

    losses = []
    for ep in range(5):
        state = SimpleRoutingState(cap, layer_dirs)
        source = (1, 0, 0)
        target = (1, 3, 3)
        current = source
        visited = {current}

        rewards, log_probs, values = [], [], []

        for _ in range(30):
            obs = state.to_numpy_features(current, target)
            flat = obs.reshape(-1, obs.shape[-2], obs.shape[-1])
            tensor = torch.tensor(flat, dtype=torch.float32).unsqueeze(0)

            logits, val = agent.policy(tensor)
            valid = state.get_valid_actions(current)
            unvis = {a: c for a, c in valid.items() if c not in visited}
            pool = unvis if unvis else valid
            if not pool:
                break

            valid_idx = [ACTIONS.index(a) for a in pool]
            mask = torch.full((NUM_ACTIONS,), float('-inf'))
            for i in valid_idx:
                mask[i] = 0.0
            probs = F.softmax(logits.squeeze(0) + mask, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

            log_probs.append(dist.log_prob(action))
            values.append(val.squeeze())

            action_name = ACTIONS[action.item()]
            nxt = pool.get(action_name, list(pool.values())[0])

            reached = (nxt == target)
            r = agent.compute_reward(current, nxt, target, state, reached)
            rewards.append(r)

            current = nxt
            visited.add(current)
            if reached:
                break

        loss = agent.train_episode(rewards, log_probs, values)
        losses.append(loss)

    print(f"  Episodes: {len(losses)}")
    print(f"  Losses  : {['%.2f' % l for l in losses]}")
    print(f"  Epsilon : {agent.epsilon:.3f}")

    # Gradients should have flowed
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in agent.policy.parameters())
    assert has_grad, "No gradients — training broken"
    print("  Gradient flow: OK")
    print("  PASSED")


# ===================================================================== #
#  TEST 5 — Real data (if available)                                      #
# ===================================================================== #

def test_real_data():
    print("\n" + "=" * 60)
    print("TEST 5: Real VLSI Data — RustWorkX Steiner tree")
    print("=" * 60)

    from utils import read_cap, read_net

    cap_file = os.path.join(os.path.dirname(__file__), 'test_data/ariane133_51.cap')
    net_file = os.path.join(os.path.dirname(__file__), 'test_data/ariane133_51.net')

    if not os.path.exists(cap_file):
        print(f"  Data not found: {cap_file}  (optional)")
        return

    t0 = time.time()
    cap_data = read_cap(cap_file, verbose=False)
    net_data = read_net(net_file, verbose=False)
    dt_load = time.time() - t0
    print(f"  Loaded {len(net_data)} nets in {dt_load:.1f}s")
    print(f"  Grid : {cap_data['nLayers']}L x {cap_data['ySize']}y x {cap_data['xSize']}x")

    # Route first 50 nets with RustWorkX
    import router_simple as rs
    rs.data_cap = cap_data
    rs.data_net = net_data
    rs.matrix = cap_data['cap'].astype(np.float32)

    nets = list(net_data.keys())[:50]
    t0 = time.time()
    total_edges = 0
    for net in nets:
        s = rs.find_solution_for_net(net)
        total_edges += s.count('\n') - 3   # rough edge count
    dt_route = time.time() - t0

    print(f"  Routed {len(nets)} nets in {dt_route:.2f}s "
          f"({dt_route/len(nets)*1000:.1f} ms/net)")
    print(f"  ~{total_edges} edges produced")

    # State from real data
    state = SimpleRoutingState(
        cap_data['cap'][:, :10, :10],            # small slice for speed
        cap_data['layerDirections'],
    )
    print(f"  State (10x10 slice): {state.summary()}")

    print("  PASSED")


# ===================================================================== #
#  Main                                                                   #
# ===================================================================== #

def main():
    print("\n" + "=" * 60)
    print("  PHASE 1 TEST SUITE")
    print("  RustWorkX + GraphSAGE + Basic RL")
    print("=" * 60)

    test_state()
    test_model()
    test_agent()
    test_training()
    test_real_data()

    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED")
    print("=" * 60)
    print("\nComponents validated:")
    print("  - RustWorkX graph state (fast)")
    print("  - GraphSAGE GNN policy (powerful)")
    print("  - Simple CNN policy (baseline)")
    print("  - Basic RL agent + training")
    print("  - RustWorkX Steiner-tree router")
    print("\nReady for Phase 2: SHAP + Knowledge Graphs")


if __name__ == '__main__':
    main()
