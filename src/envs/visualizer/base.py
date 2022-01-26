import numpy as np
from typing import Optional





class VisualizerGrid:
    def __init__(self, env, width: int, height: int, style_agent : dict, obstacles: Optional[np.ndarray]):

        self.env = env
        self.name_env = env.name
        self.style_agent = style_agent
        self.width, self.height = width, height
        if obstacles is None:
            self.obstacles = []
        else:
            assert len(obstacles.shape) == 3 and obstacles.shape[1] == obstacles.shape[2] == 2
            self.obstacles = obstacles
    
    def init_batch(self, num_episodes):
        raise NotImplementedError

    def add_viz_state(self, *args):
        raise NotImplementedError

    def render(self, *args, **kwargs):

        raise NotImplementedError

    def interact(self):
        raise NotImplementedError

    def end(self):
        raise NotImplementedError

class VisualizerGridNull(VisualizerGrid):
    def __init__(self, env, width: int, height: int, style_agent : dict, obstacles: Optional[np.ndarray]=None):
        super().__init__(env, width, height, style_agent, obstacles)

    def init_batch(self, num_episodes):
        pass

    def add_viz_state(self, *args):
        pass

    def render(self, *args, **kwargs):
        pass

    def interact(self):
        pass

    def end(self):
        pass
