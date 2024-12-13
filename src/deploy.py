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
from conditional_features import StateVectorFeatures, CNNFeatures, TimeFeatures, StepSizeFeatures
from itertools import chain
from torch.utils.tensorboard import SummaryWriter
from policy import DiffusionPolicy
from environment import make_car_racing, episode_with_dp, parallel_episodes_with_dp
from algos import DDPMAlgo, DDIMAlgo, FlowMatchingAlgo, FlowMatchingQuadraticAlgo

# =====================================================
# SETTINGS ============================================

# If more than one env, parallel execution is used to collect performance data.
# If a single env is used, can choose to save a video or gif
# note that this has additional dependencies.
NUM_ENVS = 1

# parallel env settings
# for detailed performance data use 10x10 envs or edit end of this file.
EPISODES_PER_PARALLEL_ENV = 10

# single env settings
SEED = None # to compare on specific track
SAVE_VIDEO = False
SAVE_GIF = False # requires imageio and pygifsicle

# this will load in a specific algorithm that has been trained
cfgfile = './output_flow/config.yml'

# load config
with open(cfgfile, 'r') as file:
    cfg = yaml.safe_load(file)
cfg = Box(cfg)

# If you want to override something, do it here. For example:
# cfg.STEP_SIZE = int(cfg.T / 10)

# =====================================================

K = cfg.K
device = torch.device("cuda")

# set up feature extractors
if len(cfg.STATE_SHAPE) == 3:
    state_extractor = CNNFeatures(initial_side=cfg.STATE_SHAPE[0], initial_channels=cfg.STATE_SHAPE[-1]*cfg.STATE_HISTORY, 
        layers=4, channels=48, feature_size=K).to(device)
else:
    state_extractor = StateVectorFeatures(initial_size=cfg.STATE_SHAPE[0]*cfg.STATE_HISTORY, feature_size=K).to(device)

time_extractor = TimeFeatures(times=cfg.T, feature_size=K).to(device)
if cfg.SHORTCUT:
    step_size_extractor = StepSizeFeatures(options=8, feature_size=cfg.K).to(device)


if cfg.BACKBONE == "MLP":
    pass
elif cfg.BACKBONE == "UNET":
    model = UNetBackbone(
        action_size = cfg.ACTION_SIZE, 
        action_sequence_length = cfg.NUM_ACTIONS, 
        time_condition_dimension = K, 
        states_condition_dimension = K, 
        step_size_condition_dimension = cfg.K if cfg.SHORTCUT else None,
        filters = [K, K, K, K],
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

    if cfg.SHORTCUT:
        ema_step_extractor = torch.optim.swa_utils.AveragedModel(step_size_extractor, 
            multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999))


# =====================================================
# load models

ema_model.load_state_dict(torch.load(cfg.OUTPUT_DIR+"/ema_ddpm_model.pt", weights_only=True))
ema_state_extractor.load_state_dict(torch.load(cfg.OUTPUT_DIR+"/ema_state_ex_model.pt", weights_only=True))
ema_time_extractor.load_state_dict(torch.load(cfg.OUTPUT_DIR+"/ema_tf_model.pt", weights_only=True))

if cfg.SHORTCUT:
    ema_step_extractor.load_state_dict(torch.load(cfg.OUTPUT_DIR+"/ema_stepf_model.pt", weights_only=True))

# =====================================================

if cfg.ALGO == "DDPM":
    algo = DDPMAlgo(cfg.T)
elif cfg.ALGO == "DDIM":
    algo = DDIMAlgo(cfg.T, cfg.STEP_SIZE)
elif cfg.ALGO == "FLOW":
    algo = FlowMatchingAlgo(cfg.T, cfg.STEP_SIZE)
elif cfg.ALGO == "FLOWQ":
    algo = FlowMatchingQuadraticAlgo(cfg.T, cfg.STEP_SIZE)
else:
    raise ValueError("cfg.ALGO must be one of [DDPM, DDIM, FLOW, FLOWQ]")

# =====================================================

# set up a testing environment or several
# use one for visualization and capturing video
# use multiple for collecting lots of performance data

if NUM_ENVS > 1:
    envs = []
    for i in range(NUM_ENVS):
        env = make_car_racing(cfg.STATE_SHAPE[0])
        envs.append(env)
else:
    env = make_car_racing(cfg.STATE_SHAPE[0], 0)

# set up policy, for testing
policy = DiffusionPolicy(
    state_shape=cfg.STATE_SHAPE,
    action_size=cfg.ACTION_SIZE, 
    model=ema_model if cfg.EMA else model, 
    state_extractor=ema_state_extractor if cfg.EMA else state_extractor, 
    time_extractor=ema_time_extractor if cfg.EMA else time_extractor, 
    step_size_extractor=ema_step_extractor if cfg.SHORTCUT else None, 
    device=device, 
    T=cfg.T,
    history_size=cfg.STATE_HISTORY, 
    action_sequence_length=cfg.NUM_ACTIONS, 
    sampling_algo=algo,
    channels_are_actions=cfg.CHANNELS_ARE_ACTIONS,
    clip = [np.array([-1.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])],
    parallel_envs=(NUM_ENVS>1)
)

# =====================================================


# test on environment
rewards = []
lengths = []

if NUM_ENVS > 1:
    for i in range(EPISODES_PER_PARALLEL_ENV):
        rews, lens = parallel_episodes_with_dp(envs, policy, maxlen=1000)
        rewards += rews
        lengths += lens

else:
    for i in range(1):
        st = time.time()
        r, length, frames = episode_with_dp(env, policy, maxlen=1000, 
            renders=True, return_frames=True, seed=SEED)
        rewards.append(r)
        lengths.append(length)
        print(i, "    Episode Result:", r)
        fps = (time.time()-st)/length
        print("FPS:", fps)

        # save video
        if SAVE_VIDEO:
            w, h, c = frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
            out = cv2.VideoWriter("./latest_video.mp4", fourcc, 20, (h, w))

            for image in frames:
                out.write(image)
                cv2.imshow('frame',image)
                cv2.waitKey(1)

            out.release()

        if SAVE_GIF:

            import imageio
            from pygifsicle import optimize

            imageio.mimsave('./latest_animation.gif', [f[:,:,::-1] for f in frames], fps=30, loop=0)
            optimize('./latest_animation.gif', colors=64)
            pickle.dump( frames, open( "latest_frames.p", "wb" ) )
            print("\n ========== \n")
            print("Saved ./latest_animation.gif, and pickled frames in latest_frames.p")
            print("\n ========== \n")

print("\n ========== \n")
print("RESULTS:")

print("Average Return:", np.mean(rewards), "+-", np.std(rewards))
print("Average Length:", np.mean(lengths))

# statistics if we have gathered 100 samples (10x10 environments)
if len(rewards)==100:
    rewards.sort()
    print("Average Return Middle:", np.mean(rewards[25:75]), "+-", np.std(rewards[25:75]))
    print("Average Return Top 90:", np.mean(rewards[10:]), "+-", np.std(rewards[10:]))

print("REWARDS:", rewards)
print("LENGTHS:", lengths)