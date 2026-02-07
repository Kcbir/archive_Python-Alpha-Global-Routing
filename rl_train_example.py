# coding: utf-8
"""
Example Training Script for VLSI Routing with Reinforcement Learning

Demonstrates how to use the RL framework for training agents on VLSI routing tasks.
Includes examples for:
1. Basic environment usage
2. Training with simple agents
3. Training with GNN-based agents
4. Evaluation and visualization
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import argparse
import os

from rl_environment import VLSIRoutingEnv, create_vlsi_env_from_files, RandomAgent, GreedyAgent
from rl_reward import RewardConfig, RewardCalculator, AdaptiveRewardCalculator
from rl_gnn_models import create_model, GCNRoutingPolicy, CNNRoutingPolicy
from utils import read_cap, read_net


class DQNAgent:
    """
    Deep Q-Network agent for routing
    """
    
    def __init__(self,
                 model,
                 num_actions: int = 6,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 learning_rate: float = 0.001,
                 memory_size: int = 10000):
        """
        Initialize DQN agent
        
        Args:
            model: Neural network model (Q-network)
            num_actions: Number of possible actions
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Epsilon decay rate
            learning_rate: Learning rate for optimizer
            memory_size: Size of replay buffer
        """
        self.model = model
        self.target_model = type(model)(**{k: v for k, v in model.__dict__.items() if not k.startswith('_')})
        self.target_model.load_state_dict(model.state_dict())
        self.target_model.eval()
        
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=memory_size)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.target_model.to(self.device)
    
    def select_action(self, observation, valid_actions):
        """
        Select action using epsilon-greedy policy
        
        Args:
            observation: Current state observation
            valid_actions: List of valid action strings
            
        Returns:
            action: Selected action index
        """
        # Map valid action strings to indices
        action_map = {
            'L': 0, 'R': 1, 'U': 2, 'D': 3,
            'UP_LAYER': 4, 'DOWN_LAYER': 5,
        }
        valid_action_indices = [action_map[a] for a in valid_actions if a in action_map]
        
        if not valid_action_indices:
            return 0
        
        # Epsilon-greedy
        if random.random() < self.epsilon:
            return random.choice(valid_action_indices)
        
        # Get Q-values from model
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            
            if isinstance(self.model, CNNRoutingPolicy):
                q_values, _ = self.model(obs_tensor)
            else:
                # For GNN models, need graph data
                # Simplified: use CNN branch or random
                return random.choice(valid_action_indices)
            
            q_values = q_values.cpu().numpy()[0]
            
            # Mask invalid actions
            masked_q = np.full(self.num_actions, -np.inf)
            for idx in valid_action_indices:
                masked_q[idx] = q_values[idx]
            
            return np.argmax(masked_q)
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def train_step(self, batch_size: int = 32):
        """
        Perform one training step
        
        Args:
            batch_size: Batch size for training
            
        Returns:
            loss: Training loss
        """
        if len(self.memory) < batch_size:
            return 0.0
        
        # Sample batch
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q-values
        if isinstance(self.model, CNNRoutingPolicy):
            current_q, _ = self.model(states)
        else:
            current_q = self.model(states)
        
        current_q = current_q.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Next Q-values from target network
        with torch.no_grad():
            if isinstance(self.target_model, CNNRoutingPolicy):
                next_q, _ = self.target_model(next_states)
            else:
                next_q = self.target_model(next_states)
            
            next_q_max = next_q.max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q_max
        
        # Loss
        loss = nn.MSELoss()(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network with current model weights"""
        self.target_model.load_state_dict(self.model.state_dict())
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


def train_dqn(env: VLSIRoutingEnv,
              agent: DQNAgent,
              num_episodes: int = 1000,
              batch_size: int = 32,
              target_update_freq: int = 10,
              verbose: bool = True):
    """
    Train DQN agent on routing environment
    
    Args:
        env: VLSI routing environment
        agent: DQN agent
        num_episodes: Number of training episodes
        batch_size: Batch size for training
        target_update_freq: Frequency of target network updates
        verbose: Whether to print progress
    """
    episode_rewards = []
    episode_lengths = []
    losses = []
    
    for episode in range(num_episodes):
        # Reset for each net
        net_idx = episode % len(env.net_names)
        obs = env.reset(net_idx)
        
        episode_reward = 0
        episode_loss = 0
        step_count = 0
        done = False
        
        while not done:
            # Select action
            valid_actions = env.state.get_valid_actions(env.current_position)
            action = agent.select_action(obs, valid_actions)
            
            # Take step
            next_obs, reward, done, info = env.step(action)
            
            # Store transition
            agent.store_transition(obs, action, reward, next_obs, done)
            
            # Train
            loss = agent.train_step(batch_size)
            episode_loss += loss
            
            obs = next_obs
            episode_reward += reward
            step_count += 1
            
            if done:
                break
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Update target network
        if episode % target_update_freq == 0:
            agent.update_target_network()
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(step_count)
        losses.append(episode_loss / max(step_count, 1))
        
        if verbose and episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_length = np.mean(episode_lengths[-10:])
            avg_loss = np.mean(losses[-10:])
            print(f"Episode {episode}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Avg Length: {avg_length:.1f} | "
                  f"Avg Loss: {avg_loss:.4f} | "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'losses': losses,
    }


def evaluate_agent(env: VLSIRoutingEnv, agent, num_episodes: int = 10, verbose: bool = True):
    """
    Evaluate trained agent
    
    Args:
        env: VLSI routing environment
        agent: Trained agent
        num_episodes: Number of evaluation episodes
        verbose: Whether to print results
        
    Returns:
        results: Dictionary with evaluation metrics
    """
    # Disable exploration
    if hasattr(agent, 'epsilon'):
        original_epsilon = agent.epsilon
        agent.epsilon = 0.0
    
    success_count = 0
    total_rewards = []
    total_path_lengths = []
    
    for episode in range(num_episodes):
        net_idx = episode % len(env.net_names)
        obs = env.reset(net_idx)
        
        episode_reward = 0
        done = False
        
        while not done:
            valid_actions = env.state.get_valid_actions(env.current_position)
            action = agent.select_action(obs, valid_actions)
            
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            
            if done:
                if info.get('terminal_reason') == 'target_reached':
                    success_count += 1
                    total_path_lengths.append(info.get('path_length', 0))
                break
        
        total_rewards.append(episode_reward)
        
        if verbose:
            print(f"Episode {episode}: Net {env.current_net_name} | "
                  f"Result: {info.get('terminal_reason')} | "
                  f"Reward: {episode_reward:.2f}")
    
    # Restore epsilon
    if hasattr(agent, 'epsilon'):
        agent.epsilon = original_epsilon
    
    results = {
        'success_rate': success_count / num_episodes,
        'avg_reward': np.mean(total_rewards),
        'avg_path_length': np.mean(total_path_lengths) if total_path_lengths else 0,
        'num_successes': success_count,
    }
    
    if verbose:
        print("\n=== Evaluation Results ===")
        print(f"Success Rate: {results['success_rate']:.2%}")
        print(f"Average Reward: {results['avg_reward']:.2f}")
        print(f"Average Path Length: {results['avg_path_length']:.1f}")
    
    return results


def example_basic_usage():
    """Example 1: Basic environment usage"""
    print("=== Example 1: Basic Environment Usage ===\n")
    
    # Load data
    cap_file = 'test_data/ariane133_51.cap'
    net_file = 'test_data/ariane133_51.net'
    
    if not os.path.exists(cap_file):
        print(f"Data files not found. Please ensure {cap_file} and {net_file} exist.")
        return
    
    # Create environment
    env = create_vlsi_env_from_files(cap_file, net_file)
    
    print(f"Environment created with {len(env.net_names)} nets")
    print(f"Grid size: {env.state.nLayers} layers, {env.state.ySize}x{env.state.xSize}")
    
    # Test with random agent
    agent = RandomAgent()
    obs = env.reset(0)
    
    print(f"\nRouting net: {env.current_net_name}")
    print(f"Source: {env.source}")
    print(f"Target: {env.target}")
    
    for step in range(20):
        valid_actions = env.state.get_valid_actions(env.current_position)
        if not valid_actions:
            print("No valid actions - dead end")
            break
        
        action = agent.select_action(obs, valid_actions)
        obs, reward, done, info = env.step(action)
        
        print(f"Step {step}: Action {env.action_to_str[action]} | "
              f"Position {env.current_position} | Reward {reward:.2f}")
        
        if done:
            print(f"\nTerminated: {info['terminal_reason']}")
            break


def example_train_dqn():
    """Example 2: Train DQN agent"""
    print("\n=== Example 2: Train DQN Agent ===\n")
    
    # Load data
    cap_file = 'test_data/ariane133_51.cap'
    net_file = 'test_data/ariane133_51.net'
    
    if not os.path.exists(cap_file):
        print(f"Data files not found. Skipping training example.")
        return
    
    # Create environment with custom rewards
    reward_config = RewardConfig()
    env = create_vlsi_env_from_files(
        cap_file, net_file,
        reward_config=reward_config,
        max_steps_per_net=500
    )
    
    print(f"Created environment for training")
    
    # Create CNN-based model (simpler than GNN for demo)
    model = CNNRoutingPolicy(
        input_channels=7,
        num_actions=6,
        hidden_dim=128
    )
    
    print(f"Created model: {model.__class__.__name__}")
    
    # Create DQN agent
    agent = DQNAgent(
        model=model,
        num_actions=6,
        learning_rate=0.001,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.995
    )
    
    print(f"Created agent: DQN")
    print(f"Starting training...")
    
    # Train
    results = train_dqn(
        env=env,
        agent=agent,
        num_episodes=100,
        batch_size=32,
        target_update_freq=10,
        verbose=True
    )
    
    print("\n=== Training Complete ===")
    print(f"Final average reward: {np.mean(results['episode_rewards'][-10:]):.2f}")
    
    # Evaluate
    print("\n=== Evaluation ===")
    eval_results = evaluate_agent(env, agent, num_episodes=10, verbose=True)
    
    return agent, results


def example_compare_agents():
    """Example 3: Compare different agents"""
    print("\n=== Example 3: Compare Agents ===\n")
    
    cap_file = 'test_data/ariane133_51.cap'
    net_file = 'test_data/ariane133_51.net'
    
    if not os.path.exists(cap_file):
        print(f"Data files not found. Skipping comparison.")
        return
    
    # Create environment
    env = create_vlsi_env_from_files(cap_file, net_file)
    
    # Test different agents
    agents = {
        'Random': RandomAgent(),
        'Greedy': GreedyAgent(target=env.target),
    }
    
    results = {}
    for name, agent in agents.items():
        print(f"\nTesting {name} agent...")
        result = evaluate_agent(env, agent, num_episodes=5, verbose=False)
        results[name] = result
        print(f"{name}: Success rate {result['success_rate']:.2%}, "
              f"Avg reward {result['avg_reward']:.2f}")
    
    return results


def main():
    """Main function to run examples"""
    parser = argparse.ArgumentParser(description='Train RL agent for VLSI routing')
    parser.add_argument('--example', type=int, default=1, choices=[1, 2, 3],
                       help='Which example to run (1: basic, 2: train, 3: compare)')
    parser.add_argument('--cap', type=str, default='test_data/ariane133_51.cap',
                       help='Path to .cap file')
    parser.add_argument('--net', type=str, default='test_data/ariane133_51.net',
                       help='Path to .net file')
    
    args = parser.parse_args()
    
    if args.example == 1:
        example_basic_usage()
    elif args.example == 2:
        example_train_dqn()
    elif args.example == 3:
        example_compare_agents()


if __name__ == '__main__':
    main()
