# coding: utf-8
"""
Visualization utilities for VLSI Routing RL Framework

Provides tools to visualize:
1. Routing grid states
2. Path progressions
3. Congestion heatmaps
4. Training metrics
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from typing import List, Tuple, Optional, Dict
import os


class RoutingVisualizer:
    """Visualizer for routing states and paths"""
    
    def __init__(self, state, figsize=(12, 8)):
        """
        Initialize visualizer
        
        Args:
            state: RoutingGraphState object
            figsize: Figure size for plots
        """
        self.state = state
        self.figsize = figsize
        
        # Color scheme
        self.colors = {
            'free': '#E8F4F8',          # Light blue - free cells
            'blocked': '#2C3E50',        # Dark gray - blocked
            'occupied': '#E74C3C',       # Red - occupied
            'source': '#27AE60',         # Green - source
            'target': '#3498DB',         # Blue - target
            'path': '#F39C12',           # Orange - current path
            'low_congestion': '#2ECC71', # Green - low congestion
            'high_congestion': '#E74C3C', # Red - high congestion
        }
    
    def plot_layer(self,
                   layer: int,
                   current_path: Optional[List[Tuple]] = None,
                   source: Optional[Tuple] = None,
                   target: Optional[Tuple] = None,
                   show_congestion: bool = True,
                   ax=None) -> plt.Figure:
        """
        Plot a single layer of the routing grid
        
        Args:
            layer: Layer index to plot
            current_path: Current routing path
            source: Source node (layer, y, x)
            target: Target node (layer, y, x)
            show_congestion: Whether to show congestion levels
            ax: Matplotlib axis (creates new if None)
            
        Returns:
            figure: Matplotlib figure
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        else:
            fig = ax.figure
        
        ySize, xSize = self.state.ySize, self.state.xSize
        
        # Create grid visualization
        grid = np.zeros((ySize, xSize, 3))
        
        # Base colors based on capacity and congestion
        for y in range(ySize):
            for x in range(xSize):
                if self.state.capacity_matrix[layer, y, x] == 0:
                    # Blocked
                    grid[y, x] = self._hex_to_rgb(self.colors['blocked'])
                elif show_congestion:
                    # Color by congestion
                    congestion = self.state._calculate_congestion(layer, y, x)
                    grid[y, x] = self._congestion_to_color(congestion)
                else:
                    # Free
                    if self.state.occupancy_matrix[layer, y, x] > 0:
                        grid[y, x] = self._hex_to_rgb(self.colors['occupied'])
                    else:
                        grid[y, x] = self._hex_to_rgb(self.colors['free'])
        
        # Overlay path
        if current_path:
            for node in current_path:
                if node[0] == layer:
                    y, x = node[1], node[2]
                    grid[y, x] = self._hex_to_rgb(self.colors['path'])
        
        # Mark source and target
        if source and source[0] == layer:
            y, x = source[1], source[2]
            grid[y, x] = self._hex_to_rgb(self.colors['source'])
        
        if target and target[0] == layer:
            y, x = target[1], target[2]
            grid[y, x] = self._hex_to_rgb(self.colors['target'])
        
        # Plot
        ax.imshow(grid, interpolation='nearest', origin='lower')
        ax.set_title(f'Layer {layer} - Direction: {"Horizontal" if self.state.layer_directions[layer] == 0 else "Vertical"}',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        
        # Grid lines
        ax.set_xticks(np.arange(-0.5, xSize, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, ySize, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        
        # Legend
        legend_elements = [
            mpatches.Patch(color=self.colors['free'], label='Free'),
            mpatches.Patch(color=self.colors['blocked'], label='Blocked'),
            mpatches.Patch(color=self.colors['occupied'], label='Occupied'),
            mpatches.Patch(color=self.colors['path'], label='Current Path'),
            mpatches.Patch(color=self.colors['source'], label='Source'),
            mpatches.Patch(color=self.colors['target'], label='Target'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))
        
        plt.tight_layout()
        return fig
    
    def plot_all_layers(self,
                       current_path: Optional[List[Tuple]] = None,
                       source: Optional[Tuple] = None,
                       target: Optional[Tuple] = None,
                       show_congestion: bool = True) -> plt.Figure:
        """Plot all layers in a grid"""
        n_layers = self.state.nLayers
        
        # Determine grid layout
        n_cols = min(3, n_layers)
        n_rows = (n_layers + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(self.figsize[0] * n_cols / 3, 
                                                          self.figsize[1] * n_rows / 2))
        
        if n_layers == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for layer in range(n_layers):
            self.plot_layer(layer, current_path, source, target, show_congestion, ax=axes[layer])
        
        # Hide unused subplots
        for i in range(n_layers, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('VLSI Routing Grid - All Layers', fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        return fig
    
    def plot_congestion_heatmap(self, layer: int, ax=None) -> plt.Figure:
        """Plot congestion heatmap for a layer"""
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        else:
            fig = ax.figure
        
        # Calculate congestion for all cells
        congestion = np.zeros((self.state.ySize, self.state.xSize))
        for y in range(self.state.ySize):
            for x in range(self.state.xSize):
                congestion[y, x] = self.state._calculate_congestion(layer, y, x)
        
        # Plot heatmap
        im = ax.imshow(congestion, cmap='RdYlGn_r', interpolation='nearest', 
                      origin='lower', vmin=0, vmax=1)
        
        ax.set_title(f'Congestion Heatmap - Layer {layer}', fontsize=14, fontweight='bold')
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Congestion Level', rotation=270, labelpad=20)
        
        plt.tight_layout()
        return fig
    
    def plot_path_3d(self,
                     path: List[Tuple],
                     source: Tuple,
                     target: Tuple) -> plt.Figure:
        """
        Plot routing path in 3D
        
        Args:
            path: List of (layer, y, x) tuples
            source: Source node
            target: Target node
            
        Returns:
            figure: 3D matplotlib figure
        """
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract coordinates
        if len(path) > 0:
            layers = [p[0] for p in path]
            ys = [p[1] for p in path]
            xs = [p[2] for p in path]
            
            # Plot path
            ax.plot(xs, ys, layers, 'o-', color=self.colors['path'], 
                   linewidth=2, markersize=4, label='Path')
        
        # Plot source and target
        ax.scatter([source[2]], [source[1]], [source[0]], 
                  c=self.colors['source'], s=200, marker='o', label='Source')
        ax.scatter([target[2]], [target[1]], [target[0]], 
                  c=self.colors['target'], s=200, marker='s', label='Target')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Layer')
        ax.set_title('3D Routing Path', fontsize=14, fontweight='bold')
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_training_metrics(self,
                             episode_rewards: List[float],
                             episode_lengths: List[int],
                             losses: List[float],
                             window: int = 10) -> plt.Figure:
        """
        Plot training metrics
        
        Args:
            episode_rewards: List of episode rewards
            episode_lengths: List of episode lengths
            losses: List of training losses
            window: Moving average window
            
        Returns:
            figure: Matplotlib figure
        """
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        episodes = np.arange(len(episode_rewards))
        
        # Plot rewards
        axes[0].plot(episodes, episode_rewards, alpha=0.3, color='blue', label='Raw')
        if len(episode_rewards) >= window:
            moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
            axes[0].plot(episodes[window-1:], moving_avg, color='red', linewidth=2, label=f'MA({window})')
        axes[0].set_ylabel('Episode Reward')
        axes[0].set_title('Training Rewards', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot episode lengths
        axes[1].plot(episodes, episode_lengths, alpha=0.3, color='green', label='Raw')
        if len(episode_lengths) >= window:
            moving_avg = np.convolve(episode_lengths, np.ones(window)/window, mode='valid')
            axes[1].plot(episodes[window-1:], moving_avg, color='orange', linewidth=2, label=f'MA({window})')
        axes[1].set_ylabel('Episode Length')
        axes[1].set_title('Episode Lengths', fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot losses
        axes[2].plot(episodes, losses, alpha=0.3, color='purple', label='Raw')
        if len(losses) >= window:
            moving_avg = np.convolve(losses, np.ones(window)/window, mode='valid')
            axes[2].plot(episodes[window-1:], moving_avg, color='brown', linewidth=2, label=f'MA({window})')
        axes[2].set_xlabel('Episode')
        axes[2].set_ylabel('Loss')
        axes[2].set_title('Training Loss', fontsize=12, fontweight='bold')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.suptitle('Training Metrics', fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def plot_state_summary(self) -> plt.Figure:
        """Plot summary statistics of current state"""
        summary = self.state.get_state_summary()
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        # Utilization
        axes[0, 0].bar(['Utilization'], [summary['utilization']], color=self.colors['occupied'])
        axes[0, 0].set_ylim([0, 1])
        axes[0, 0].set_ylabel('Ratio')
        axes[0, 0].set_title('Capacity Utilization')
        axes[0, 0].text(0, summary['utilization'] + 0.05, f"{summary['utilization']:.2%}", 
                       ha='center', fontweight='bold')
        
        # Routed nets
        axes[0, 1].bar(['Routed Nets'], [summary['num_routed_nets']], color=self.colors['source'])
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Number of Routed Nets')
        axes[0, 1].text(0, summary['num_routed_nets'] + 0.5, str(summary['num_routed_nets']),
                       ha='center', fontweight='bold')
        
        # Average congestion
        axes[1, 0].bar(['Avg Congestion'], [summary['avg_congestion']], 
                      color=self._congestion_to_color(summary['avg_congestion']))
        axes[1, 0].set_ylim([0, 1])
        axes[1, 0].set_ylabel('Level')
        axes[1, 0].set_title('Average Congestion')
        axes[1, 0].text(0, summary['avg_congestion'] + 0.05, f"{summary['avg_congestion']:.2%}",
                       ha='center', fontweight='bold')
        
        # Blocked cells
        total_cells = self.state.nLayers * self.state.ySize * self.state.xSize
        blocked_ratio = summary['blocked_cells'] / total_cells
        axes[1, 1].bar(['Blocked Cells'], [blocked_ratio], color=self.colors['blocked'])
        axes[1, 1].set_ylim([0, 1])
        axes[1, 1].set_ylabel('Ratio')
        axes[1, 1].set_title('Blocked Cells Ratio')
        axes[1, 1].text(0, blocked_ratio + 0.05, f"{blocked_ratio:.2%}",
                       ha='center', fontweight='bold')
        
        plt.suptitle('Routing State Summary', fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def animate_routing(self,
                       path_history: List[List[Tuple]],
                       source: Tuple,
                       target: Tuple,
                       layer: int,
                       save_path: Optional[str] = None,
                       interval: int = 500):
        """
        Create animation of routing process
        
        Args:
            path_history: List of paths at each step
            source: Source node
            target: Target node
            layer: Layer to visualize
            save_path: Path to save GIF (None for display only)
            interval: Milliseconds between frames
        """
        from matplotlib.animation import FuncAnimation
        
        fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        
        def update(frame):
            ax.clear()
            path = path_history[frame] if frame < len(path_history) else path_history[-1]
            self.plot_layer(layer, path, source, target, show_congestion=True, ax=ax)
            ax.set_title(f'Layer {layer} - Step {frame + 1}/{len(path_history)}', 
                        fontsize=14, fontweight='bold')
        
        anim = FuncAnimation(fig, update, frames=len(path_history),
                           interval=interval, repeat=True)
        
        if save_path:
            anim.save(save_path, writer='pillow')
            print(f"Animation saved to {save_path}")
        else:
            plt.show()
        
        return anim
    
    def _hex_to_rgb(self, hex_color: str) -> np.ndarray:
        """Convert hex color to RGB array"""
        hex_color = hex_color.lstrip('#')
        return np.array([int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4)])
    
    def _congestion_to_color(self, congestion: float) -> np.ndarray:
        """Map congestion level to color"""
        # Green (low) -> Yellow (medium) -> Red (high)
        if congestion < 0.33:
            # Green to yellow
            t = congestion / 0.33
            return np.array([t, 1.0, 0.0])
        elif congestion < 0.67:
            # Yellow to orange
            t = (congestion - 0.33) / 0.34
            return np.array([1.0, 1.0 - t * 0.5, 0.0])
        else:
            # Orange to red
            t = (congestion - 0.67) / 0.33
            return np.array([1.0, 0.5 - t * 0.5, 0.0])
    
    def save_figure(self, fig: plt.Figure, filename: str, dpi: int = 300):
        """Save figure to file"""
        fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to {filename}")


def visualize_episode(env, agent, save_dir: Optional[str] = None):
    """
    Visualize a complete episode
    
    Args:
        env: VLSIRoutingEnv instance
        agent: Agent to use for actions
        save_dir: Directory to save visualizations (None for display only)
    """
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    obs = env.reset(0)
    visualizer = RoutingVisualizer(env.state)
    
    # Initial state
    fig = visualizer.plot_all_layers(source=env.source, target=env.target)
    if save_dir:
        visualizer.save_figure(fig, os.path.join(save_dir, 'initial_state.png'))
    else:
        plt.show()
    
    # Step through episode
    path_history = []
    step = 0
    done = False
    
    while not done and step < 100:
        valid_actions = env.state.get_valid_actions(env.current_position)
        if not valid_actions:
            break
        
        action = agent.select_action(obs, valid_actions)
        obs, reward, done, info = env.step(action)
        
        path_history.append(env.current_path.copy())
        
        # Plot every 10 steps or on completion
        if step % 10 == 0 or done:
            layer = env.current_position[0]
            fig = visualizer.plot_layer(layer, env.current_path, env.source, env.target)
            if save_dir:
                visualizer.save_figure(fig, os.path.join(save_dir, f'step_{step:03d}.png'))
            plt.close(fig)
        
        step += 1
    
    # Final 3D path
    if done and info.get('terminal_reason') == 'target_reached':
        fig = visualizer.plot_path_3d(env.current_path, env.source, env.target)
        if save_dir:
            visualizer.save_figure(fig, os.path.join(save_dir, 'final_path_3d.png'))
        else:
            plt.show()
    
    # State summary
    fig = visualizer.plot_state_summary()
    if save_dir:
        visualizer.save_figure(fig, os.path.join(save_dir, 'state_summary.png'))
    else:
        plt.show()
    
    print(f"Episode completed: {info.get('terminal_reason')}")
    return path_history


if __name__ == '__main__':
    # Example usage
    from rl_environment import create_vlsi_env_from_files, RandomAgent
    
    # Load environment
    env = create_vlsi_env_from_files(
        'test_data/ariane133_51.cap',
        'test_data/ariane133_51.net'
    )
    
    # Visualize
    agent = RandomAgent()
    visualize_episode(env, agent, save_dir='visualizations')
