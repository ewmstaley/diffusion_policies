'''
Copyright © 2024 The Johns Hopkins University Applied Physics Laboratory LLC
 
Permission is hereby granted, free of charge, to any person obtaining a copy 
of this software and associated documentation files (the “Software”), to 
deal in the Software without restriction, including without limitation the 
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or 
sell copies of the Software, and to permit persons to whom the Software is 
furnished to do so, subject to the following conditions:
 
The above copyright notice and this permission notice shall be included in 
all copies or substantial portions of the Software.
 
THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR 
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

'''
Script to run a saved diffusion policy on car racing.
'''

import torch
import numpy as np
from backbones.mlp import MLPBackbone
from backbones.unet import UNetBackbone
import cv2
import math
from tqdm import tqdm
from utility import get_diffusion_parameters, cosine_lr_scheduler
import matplotlib.pyplot as plt
import os
import pickle
from box import Box
import yaml
import glob
import time
from buffer import TrajectoryBuffer
from conditional_features import StateVectorFeatures, CNNFeatures, TimeFeatures
from itertools import chain
from torch.utils.tensorboard import SummaryWriter
from policy import DiffusionPolicy
from environment import make_car_racing, episode_with_dp


# load config
with open('./output/config.yml', 'r') as file:
    cfg = yaml.safe_load(file)
cfg = Box(cfg)

# =====================================================

device = torch.device("cuda")

# set up feature extractors
if len(cfg.STATE_SHAPE) == 3:
    state_extractor = CNNFeatures(initial_side=cfg.STATE_SHAPE[0], initial_channels=cfg.STATE_SHAPE[-1]*cfg.STATE_HISTORY, 
        layers=4, channels=48, feature_size=256).to(device)
else:
    state_extractor = StateVectorFeatures(initial_size=cfg.STATE_SHAPE[0]*cfg.STATE_HISTORY, feature_size=256).to(device)

time_extractor = TimeFeatures(times=cfg.T, feature_size=256).to(device)

if cfg.BACKBONE == "MLP":
    pass
elif cfg.BACKBONE == "UNET":
    model = UNetBackbone(
        action_size = cfg.ACTION_SIZE, 
        action_sequence_length = cfg.NUM_ACTIONS, 
        time_condition_dimension = 256, 
        states_condition_dimension = 256, 
        filters = [256, 256, 256, 256],
        pools = [True, True, False],
        channels_are_actions = cfg.CHANNELS_ARE_ACTIONS
    ).to(device)

if cfg.EMA:
    ema_model = torch.optim.swa_utils.AveragedModel(model, 
        multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999))
    ema_state_extractor = torch.optim.swa_utils.AveragedModel(state_extractor, 
        multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999))
    ema_time_extractor = torch.optim.swa_utils.AveragedModel(time_extractor, 
        multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999))

# =====================================================
# load models

ema_model.load_state_dict(torch.load(cfg.OUTPUT_DIR+"/ema_ddpm_model.pt", weights_only=True))
ema_state_extractor.load_state_dict(torch.load(cfg.OUTPUT_DIR+"/ema_state_ex_model.pt", weights_only=True))
ema_time_extractor.load_state_dict(torch.load(cfg.OUTPUT_DIR+"/ema_tf_model.pt", weights_only=True))

# =====================================================

# set up a testing environment
env = make_car_racing(cfg.STATE_SHAPE[0])

# set up policy, for testing
policy = DiffusionPolicy(
    state_shape=cfg.STATE_SHAPE,
    action_size=cfg.ACTION_SIZE, 
    model=ema_model if cfg.EMA else model, 
    state_extractor=ema_state_extractor if cfg.EMA else state_extractor, 
    time_extractor=ema_time_extractor if cfg.EMA else time_extractor, 
    device=device, 
    T=cfg.T,
    history_size=cfg.STATE_HISTORY, 
    action_sequence_length=cfg.NUM_ACTIONS, 
    implicit=True, 
    sampling_factor=20,
    channels_are_actions=cfg.CHANNELS_ARE_ACTIONS,
    clip = [np.array([-1.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])]
)

# =====================================================

# test on environment and save a video
# Caution: this can take a long time to run.

rewards = []
lengths = []
for i in range(100):
    st = time.time()
    r, length, frames = episode_with_dp(env, policy, maxlen=1000, renders=True, return_frames=True)
    rewards.append(r)
    lengths.append(length)
    print(i, "    Episode Result:", r)
    fps = (time.time()-st)/length
    print("FPS:", fps)

    # save video
    w, h, c = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
    out = cv2.VideoWriter("./video2.mp4", fourcc, 20, (h, w))

    for image in frames:
        out.write(image)
        cv2.imshow('frame',image)
        cv2.waitKey(1)

    out.release()

print("Average Return:", np.mean(rewards), "+-", np.std(rewards))
print("Average Length:", np.mean(lengths))
