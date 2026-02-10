# coding: utf-8
"""
Simple RL Agent for VLSI Global Routing
GraphSAGE (powerful) + CNN (fast) policies with basic RL.

Architecture:
  Policy network  →  action logits  +  state value
  Agent           →  epsilon-greedy  +  REINFORCE training
"""

import numpy as np
import random
from collections import deque

# ---------------------------------------------------------------------------
# Action space (must match SimpleRoutingState.ACTIONS)
# ---------------------------------------------------------------------------
ACTIONS = ['L', 'R', 'U', 'D', 'UP_LAYER', 'DOWN_LAYER']
NUM_ACTIONS = len(ACTIONS)

# ---------------------------------------------------------------------------
# Optional imports — framework degrades gracefully
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from torch_geometric.nn import SAGEConv, global_mean_pool
    from torch_geometric.data import Data, Batch
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


# ===================================================================== #
#  POLICIES                                                              #
# ===================================================================== #

class GraphSAGEPolicy(nn.Module):
    """
    3-layer GraphSAGE  →  global-mean-pool  →  action / value heads.

    Why GraphSAGE?
      • Inductive: generalises to unseen circuits
      • Scalable:  samples neighbours, doesn't need full graph
      • Powerful:  state-of-the-art graph learning
    """

    def __init__(self, input_dim=8, hidden_dim=64, num_actions=NUM_ACTIONS):
        super().__init__()
        if not HAS_PYG:
            raise ImportError("torch-geometric is required for GraphSAGEPolicy")

        self.sage1 = SAGEConv(input_dim, hidden_dim)
        self.sage2 = SAGEConv(hidden_dim, hidden_dim)
        self.sage3 = SAGEConv(hidden_dim, hidden_dim)

        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data):
        """
        Args
        ----
        data : torch_geometric.data.Data  (or Batch)
            Must have ``.x`` and ``.edge_index``.

        Returns
        -------
        action_logits : (batch, NUM_ACTIONS)
        value         : (batch, 1)
        """
        x, ei = data.x, data.edge_index
        batch = (data.batch if hasattr(data, 'batch') and data.batch is not None
                 else torch.zeros(x.size(0), dtype=torch.long, device=x.device))

        x = F.relu(self.sage1(x, ei))
        x = F.relu(self.sage2(x, ei))
        x = F.relu(self.sage3(x, ei))

        g = global_mean_pool(x, batch)          # (B, hidden)
        return self.action_head(g), self.value_head(g)


class SimpleCNNPolicy(nn.Module):
    """
    Lightweight 2-D CNN policy (fast baseline).

    Input : (batch, channels, H, W)
    Output: action_logits, value
    """

    def __init__(self, input_channels=10, hidden_dim=64, num_actions=NUM_ACTIONS):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.pool  = nn.AdaptiveAvgPool2d(1)

        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        """x : (batch, C, H, W)"""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x).flatten(1)             # (batch, hidden)
        return self.action_head(x), self.value_head(x)


# ===================================================================== #
#  RL AGENT                                                              #
# ===================================================================== #

class SimpleRLAgent:
    """
    Epsilon-greedy actor-critic agent.

    Supports both GNN (GraphSAGE) and CNN backends.
    Training via REINFORCE with baseline (simple & effective).
    """

    def __init__(self, policy, use_gnn=False, lr=1e-3, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995):
        self.policy = policy
        self.use_gnn = use_gnn
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        if HAS_TORCH:
            self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

        self.memory = deque(maxlen=10000)

    # ---------------------------------------------------------------- #
    #  Action selection                                                  #
    # ---------------------------------------------------------------- #
    def select_action(self, obs, valid_actions):
        """
        Epsilon-greedy action selection.

        Parameters
        ----------
        obs : np.ndarray | torch_geometric.data.Data
            Observation (numpy features for CNN, PyG data for GNN).
        valid_actions : dict | list
            {action_name: next_coord}  or  [action_name, ...]

        Returns
        -------
        int  – index into ACTIONS list
        """
        if isinstance(valid_actions, dict):
            valid_names = list(valid_actions.keys())
        else:
            valid_names = list(valid_actions)
        valid_idx = [ACTIONS.index(a) for a in valid_names if a in ACTIONS]
        if not valid_idx:
            return 0

        # ε-greedy exploration
        if random.random() < self.epsilon:
            return random.choice(valid_idx)

        if not HAS_TORCH:
            return random.choice(valid_idx)

        with torch.no_grad():
            if self.use_gnn and HAS_PYG and isinstance(obs, Data):
                logits, _ = self.policy(obs)
            else:
                if isinstance(obs, np.ndarray):
                    flat = obs.reshape(-1, obs.shape[-2], obs.shape[-1])
                    tensor = torch.tensor(flat, dtype=torch.float32).unsqueeze(0)
                else:
                    tensor = obs
                logits, _ = self.policy(tensor)

        logits = logits.squeeze(0)
        mask = torch.full((NUM_ACTIONS,), float('-inf'))
        for i in valid_idx:
            mask[i] = 0.0
        return (logits + mask).argmax().item()

    # ---------------------------------------------------------------- #
    #  Reward function                                                   #
    # ---------------------------------------------------------------- #
    @staticmethod
    def compute_reward(current, next_coord, target, state,
                       reached_target=False):
        """
        Simple, effective reward shaping:

            +100   reached target
             +2    moved closer  (Manhattan)
             −2    moved farther
             −1    step penalty  (prefer short paths)
             −5×u  congestion penalty  (u = usage ratio > 0.8)
            −10    overflow cell
             −0.5  via penalty  (avoid unnecessary layer hops)
            −50    dead-end / invalid
        """
        if reached_target:
            return 100.0
        if next_coord is None:
            return -50.0

        # Manhattan distance improvement
        d0 = (abs(current[0] - target[0]) + abs(current[1] - target[1])
              + abs(current[2] - target[2]))
        d1 = (abs(next_coord[0] - target[0]) + abs(next_coord[1] - target[1])
              + abs(next_coord[2] - target[2]))

        reward = -1.0                                   # step cost
        reward += 2.0 if d1 < d0 else (-2.0 if d1 > d0 else 0.0)

        # Congestion
        z, y, x = next_coord
        cap  = state.cap[z, y, x]
        orig = state.original_cap[z, y, x]
        if orig > 0:
            usage = 1.0 - cap / orig
            if usage > 0.8:
                reward -= 5.0 * usage
            if cap <= 0:
                reward -= 10.0

        # Via penalty
        if current[0] != next_coord[0]:
            reward -= 0.5

        return reward

    # ---------------------------------------------------------------- #
    #  Training (REINFORCE with baseline)                                #
    # ---------------------------------------------------------------- #
    def train_episode(self, rewards, log_probs, values):
        """
        One gradient step from a completed episode.

        Returns loss value (float).
        """
        if not HAS_TORCH or not rewards:
            return 0.0

        # Discounted returns
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        policy_loss = torch.tensor(0.0)
        value_loss  = torch.tensor(0.0)
        for lp, v, G in zip(log_probs, values, returns):
            adv = G - v.detach().squeeze()
            policy_loss = policy_loss - lp * adv
            value_loss  = value_loss + F.smooth_l1_loss(v.squeeze(), G)

        loss = policy_loss + 0.5 * value_loss
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

        self.epsilon = max(self.epsilon_end,
                           self.epsilon * self.epsilon_decay)
        return loss.item()

    # ---------------------------------------------------------------- #
    #  Complete net routing (RL walk)                                     #
    # ---------------------------------------------------------------- #
    def route_net_rl(self, state, source, target, max_steps=200):
        """
        Route a 2-pin net by stepping from *source* to *target*.

        Returns
        -------
        path          : list[(z, y, x)]
        total_reward  : float
        success       : bool
        """
        current = source
        path = [current]
        total_reward = 0.0
        visited = {current}

        for _ in range(max_steps):
            valid = state.get_valid_actions(current)

            # Prefer unvisited cells to avoid loops
            unvisited = {a: c for a, c in valid.items() if c not in visited}
            pool = unvisited if unvisited else valid
            if not pool:
                break

            # Pick action
            if self.use_gnn and HAS_PYG:
                obs = state.to_pyg_data(current, target)
            else:
                obs = state.to_numpy_features(current, target)

            action_idx = self.select_action(obs, pool)
            action_name = ACTIONS[action_idx]

            nxt = pool.get(action_name)
            if nxt is None:                          # fallback: any valid
                action_name = list(pool.keys())[0]
                nxt = pool[action_name]

            reached = (nxt == target)
            reward = self.compute_reward(current, nxt, target, state, reached)
            total_reward += reward

            current = nxt
            path.append(current)
            visited.add(current)

            if reached:
                return path, total_reward, True

        return path, total_reward, False


# ===================================================================== #
#  Factory                                                               #
# ===================================================================== #

def create_simple_agent(use_gnn=False, input_channels=10, input_dim=8,
                        hidden_dim=64, **kwargs):
    """
    Create an RL agent with the chosen policy backend.

    Parameters
    ----------
    use_gnn        : bool   – True → GraphSAGE, False → CNN
    input_channels : int    – CNN input channels (features × nLayers)
    input_dim      : int    – GNN per-node feature dimension
    hidden_dim     : int    – hidden layer width
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch is required")
    if use_gnn:
        if not HAS_PYG:
            raise ImportError("torch-geometric is required for GraphSAGE")
        policy = GraphSAGEPolicy(input_dim=input_dim, hidden_dim=hidden_dim)
    else:
        policy = SimpleCNNPolicy(input_channels=input_channels,
                                 hidden_dim=hidden_dim)
    return SimpleRLAgent(policy, use_gnn=use_gnn, **kwargs)
