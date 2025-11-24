import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from typing import Optional, Tuple

class BreastCancerAwarenessEnv(gym.Env):
    """
    Custom Environment for Breast Cancer Awareness Campaign Optimization
    
    ABSOLUTE SIMPLEST VERSION - Dense rewards everywhere
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}
    
    # Cell types
    EMPTY = 0
    WOMAN_UNAWARE = 1
    WOMAN_EDUCATED = 2
    
    def __init__(self, render_mode: Optional[str] = None, grid_size: int = 5):
        super().__init__()
        
        self.grid_size = grid_size
        self.render_mode = render_mode
        
        # Define action space: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, 4=EDUCATE
        self.action_space = spaces.Discrete(5)
        
        # Observation space
        obs_shape = 2 + (grid_size * grid_size) + 2 + 1
        self.observation_space = spaces.Box(
            low=0, 
            high=max(grid_size, 10),
            shape=(obs_shape,), 
            dtype=np.float32
        )
        
        # Initialize pygame for rendering
        self.window = None
        self.clock = None
        self.cell_size = 120
        self.window_size = self.grid_size * self.cell_size
        
        # Episode tracking
        self.max_steps = 80
        self.current_step = 0
        
        # Initialize environment state
        self.reset()
    
    def _get_obs(self) -> np.ndarray:
        """Return the current observation."""
        obs = np.concatenate([
            np.array([self.agent_pos[0], self.agent_pos[1]], dtype=np.float32),
            self.grid.flatten().astype(np.float32),
            np.array([self.women_reached, self.total_women], dtype=np.float32),
            np.array([self.current_step], dtype=np.float32)
        ])
        return obs
    
    def _get_info(self) -> dict:
        """Return additional information about the environment state."""
        return {
            'women_reached': self.women_reached,
            'total_women': self.total_women,
            'coverage_rate': self.women_reached / max(self.total_women, 1),
            'step': self.current_step
        }
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Initialize grid
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        
        # Only 3 women - in a simple line
        self.woman_positions = [
            (1, 1),  # Close
            (2, 2),  # Medium
            (3, 3)   # Far
        ]
        
        for pos in self.woman_positions:
            self.grid[pos] = self.WOMAN_UNAWARE
        
        # Initialize agent at top-left
        self.agent_pos = np.array([0, 0])
        
        # Initialize statistics
        self.women_reached = 0
        self.total_women = len(self.woman_positions)
        self.current_step = 0
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step in the environment."""
        self.current_step += 1
        reward = 10.0  # BASE reward for every action - always positive!
        terminated = False
        truncated = False
        
        # Movement actions
        if action in [0, 1, 2, 3]:
            old_pos = self.agent_pos.copy()
            new_pos = self.agent_pos.copy()
            
            if action == 0:  # UP
                new_pos[0] = max(0, self.agent_pos[0] - 1)
            elif action == 1:  # DOWN
                new_pos[0] = min(self.grid_size - 1, self.agent_pos[0] + 1)
            elif action == 2:  # LEFT
                new_pos[1] = max(0, self.agent_pos[1] - 1)
            elif action == 3:  # RIGHT
                new_pos[1] = min(self.grid_size - 1, self.agent_pos[1] + 1)
            
            # Check if actually moved
            if not np.array_equal(old_pos, new_pos):
                self.agent_pos = new_pos
                reward += 10  # Bonus for actually moving
                
                # Check current cell
                current_cell = tuple(self.agent_pos)
                if self.grid[current_cell] == self.WOMAN_UNAWARE:
                    reward += 1000  # HUGE bonus for landing on woman
                elif self.grid[current_cell] == self.WOMAN_EDUCATED:
                    reward += 50  # Still positive even if already educated
        
        # Educate action
        elif action == 4:
            current_cell = tuple(self.agent_pos)
            
            if self.grid[current_cell] == self.WOMAN_UNAWARE:
                # SUCCESS!
                self.grid[current_cell] = self.WOMAN_EDUCATED
                self.women_reached += 1
                reward += 5000  # MASSIVE reward
                
                # Milestone bonuses
                if self.women_reached == 1:
                    reward += 1000
                elif self.women_reached == 2:
                    reward += 2000
                elif self.women_reached == 3:
                    reward += 5000
                    terminated = True  # Success!
            else:
                reward += 5  # Still positive even if wrong
        
        # Time limit
        if self.current_step >= self.max_steps:
            truncated = True
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment."""
        if self.render_mode is None:
            return
        
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size + 100))
            pygame.display.set_caption("Breast Cancer Awareness Campaign")
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        
        canvas = pygame.Surface((self.window_size, self.window_size + 100))
        canvas.fill((255, 255, 255))
        
        # Colors
        COLORS = {
            self.EMPTY: (240, 255, 240),
            self.WOMAN_UNAWARE: (255, 182, 193),
            self.WOMAN_EDUCATED: (144, 238, 144)
        }
        
        # Draw grid
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                rect = pygame.Rect(
                    j * self.cell_size,
                    i * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
                cell_type = self.grid[i, j]
                pygame.draw.rect(canvas, COLORS[cell_type], rect)
                pygame.draw.rect(canvas, (100, 100, 100), rect, 3)
                
                # Draw icons
                font = pygame.font.Font(None, 70)
                if cell_type == self.WOMAN_UNAWARE:
                    text = font.render('W', True, (0, 0, 0))
                    canvas.blit(text, (j * self.cell_size + 40, i * self.cell_size + 35))
                elif cell_type == self.WOMAN_EDUCATED:
                    text = font.render('✓', True, (0, 100, 0))
                    canvas.blit(text, (j * self.cell_size + 40, i * self.cell_size + 35))
        
        # Draw agent
        pygame.draw.circle(
            canvas, 
            (0, 0, 255), 
            (self.agent_pos[1] * self.cell_size + self.cell_size // 2,
             self.agent_pos[0] * self.cell_size + self.cell_size // 2),
            self.cell_size // 3
        )
        
        # Draw stats
        font = pygame.font.Font(None, 32)
        stats_y = self.window_size + 20
        stats_text = [
            f"Women: {self.women_reached}/{self.total_women}",
            f"Coverage: {self.women_reached/self.total_women*100:.0f}%",
            f"Step: {self.current_step}/{self.max_steps}"
        ]
        
        for idx, text in enumerate(stats_text):
            text_surface = font.render(text, True, (0, 0, 0))
            canvas.blit(text_surface, (15 + idx * 180, stats_y))
        
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
        )
    
    def close(self):
        """Close the rendering window."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None