import isaacgym
import isaacgyminsertion
import torch
import numpy as np
from isaacgyminsertion.utils import torch_jit_utils

num_envs = 16
seed = 0

torch.manual_seed(seed)
np.random.seed(seed)

envs = isaacgyminsertion.make(
    seed=0, 
    task="FactoryTaskInsertionTactile", 
    num_envs=num_envs, 
    sim_device="cuda:0",
    rl_device="cuda:0",
    graphics_device_id=0,
    headless=False,
    multi_gpu=False,
    virtual_screen_capture=False,
    force_render=True,
)

print("Observation space is", envs.observation_space)
print("Action space is", envs.action_space)
obs = envs.reset()
step = 0
while envs.viewer:

    actions = torch.zeros((num_envs, 6), device="cuda:0")

    obs, rew, reset_buf, info = envs.step(actions)

    step += 1