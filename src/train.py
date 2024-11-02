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
Train a diffusion policy on expert data.
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
from buffer import TrajectoryBuffer
from conditional_features import StateVectorFeatures, CNNFeatures, TimeFeatures
from itertools import chain
from torch.utils.tensorboard import SummaryWriter
from policy import DiffusionPolicy
from environment import make_car_racing, episode_with_dp



# =====================================================
# some configuration

cfg = Box()

cfg.BACKBONE = "UNET"
cfg.ACTION_SIZE = 3
cfg.STATE_SHAPE = (96,96,3)
cfg.STATE_HISTORY = 4
cfg.NUM_ACTIONS = 8
cfg.CHANNELS_ARE_ACTIONS = False
cfg.T = 1000
cfg.BATCH_SIZE = 32
cfg.EPOCHS = 1000
cfg.ITERS_PER_EPOCH = 500
cfg.EMA = True
cfg.MIXED_PRECISION = False
cfg.DTYPE = "bfloat16"
cfg.OUTPUT_DIR = "./output/"
cfg.TESTING_MAXLEN = 1000
# =====================================================

if not os.path.exists(cfg.OUTPUT_DIR):
    os.makedirs(cfg.OUTPUT_DIR)

# save config
with open(cfg.OUTPUT_DIR+'config.yml', 'w') as outfile:
    yaml.dump(cfg.to_dict(), outfile)

# get data loaded in nice format
buffer = TrajectoryBuffer(
    shapes=[cfg.STATE_SHAPE, (cfg.ACTION_SIZE,)], 
    sequence_length=cfg.STATE_HISTORY+cfg.NUM_ACTIONS-1
)

# -- car racing data -- #
for f in glob.glob("../data_collect/excellent_runs/*.p"):
    print(f)
    tuples, reward, length = pickle.load( open( f, "rb" ) )
    states = [x["state"] for x in tuples]
    actions = [x["action"] for x in tuples]

    # normalize the states
    states = [s.astype(np.float32)/255.0 for s in states]

    # NOTE: Currently do not handle "history" for early states,
    # will add this if it seems needed.

    buffer.add_trajectory([states, actions])


# =====================================================
# set up our models. We actually have six models happening:
# the time feature extractor, the image state feature extractor, and the diffusion model
#   these are all trained together
# for each one, an EMA copy that keeps a running (weighted) average of the parameters.

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


all_params = chain(state_extractor.parameters(), time_extractor.parameters(), model.parameters())

opt = torch.optim.Adam(all_params, lr=0.0001, weight_decay=1e-6)
total_steps = cfg.ITERS_PER_EPOCH*cfg.EPOCHS
lr_sched = cosine_lr_scheduler(opt, total_steps, warmup_steps=500, final=0.001)

if cfg.EMA:
    ema_model = torch.optim.swa_utils.AveragedModel(model, 
        multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999))
    ema_state_extractor = torch.optim.swa_utils.AveragedModel(state_extractor, 
        multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999))
    ema_time_extractor = torch.optim.swa_utils.AveragedModel(time_extractor, 
        multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999))

if cfg.MIXED_PRECISION:
    scaler = torch.cuda.amp.GradScaler()

state_ex_params = sum(p.numel() for p in state_extractor.parameters())
tf_params = sum(p.numel() for p in time_extractor.parameters())
backbone_params = sum(p.numel() for p in model.parameters())
print("---")
print("State Extractor has", f'{state_ex_params:,}', "parameters.")
print("Time Extractor has", f'{tf_params:,}', "parameters.")
print("Backbone has", f'{backbone_params:,}', "parameters.")
print("---")

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

# pre-calculate coefficients as a function of time t
betas, alphas, alpha_hats = get_diffusion_parameters(cfg.T)

# =====================================================
# training loop

logger = SummaryWriter(log_dir=cfg.OUTPUT_DIR)

# train
best_loss = 100000.0
loss_history = []
batches = 0

for epoch in range(cfg.EPOCHS):
    print("EPOCH", epoch+1)

    epoch_losses = []
    for b in tqdm(range(cfg.ITERS_PER_EPOCH)):
        opt.zero_grad()
        B = cfg.BATCH_SIZE

        # get some subtrajectories
        steps = buffer.sample(B) # [[(B,S), (B,A)], [(B,S), (B,A)], [(B,S), (B,A)], ...]
        history = steps[:cfg.STATE_HISTORY]
        targets = steps[cfg.STATE_HISTORY-1:]
        history = [x[0] for x in history] # [(B,S), (B,S), ...]
        targets = [x[1] for x in targets] # [(B,A), (B,A), ...]

        history = np.stack(history, axis=-1) # (B, S, history_seq)
        if len(cfg.STATE_SHAPE)==3:
            history = np.reshape(history, (*history.shape[:3], -1)) # (B, w, h, ch*4)
            history = np.transpose(history, (0,3,1,2)) # (B, ch*4, w, h)
        else:
            history = np.reshape(history, (B, -1)) # (B, S*history_seq)
        history = torch.from_numpy(history).to(torch.float32).to(device)

        targets = np.stack(targets, axis=-1) # (B, A, seq)
        if cfg.CHANNELS_ARE_ACTIONS:
            targets = np.transpose(targets, (0,2,1)) # (B, seq, A)
        targets = torch.from_numpy(targets).to(torch.float32).to(device)

        # also sample noise, times, and alpha_hats
        e = torch.randn(*targets.shape).to(torch.float32).to(device)
        t = np.random.randint(0, cfg.T, size=(B,))
        ahat = alpha_hats[t]
        t = torch.as_tensor(t).to(torch.long).to(device)
        ahat = torch.as_tensor(ahat).to(torch.float32).to(device)

        if cfg.CHANNELS_ARE_ACTIONS:
            ahat = ahat.unsqueeze(1).unsqueeze(2).repeat(1,cfg.NUM_ACTIONS,cfg.ACTION_SIZE)
        else:
            ahat = ahat.unsqueeze(1).unsqueeze(2).repeat(1,cfg.ACTION_SIZE,cfg.NUM_ACTIONS)

        if cfg.MIXED_PRECISION:
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):

                # get the conditioning features
                state_features = state_extractor(history)
                time_features = time_extractor(t)

                # get the noised targets
                noised_batch = torch.sqrt(ahat)*targets + torch.sqrt(1.0 - ahat)*e

                # main pass
                predicted_noise = model(noised_batch, time_features, state_features)
                loss = torch.nn.functional.mse_loss(predicted_noise, e)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            scaler.step(opt)
            scaler.update()

        else:
            state_features = state_extractor(history)
            time_features = time_extractor(t)
            noised_batch = torch.sqrt(ahat)*targets + torch.sqrt(1.0 - ahat)*e
            predicted_noise = model(noised_batch, time_features, state_features)
            loss = torch.nn.functional.mse_loss(predicted_noise, e)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            opt.step()

        if not math.isnan(loss.item()):
            epoch_losses.append(loss.item())
            logger.add_scalar("loss", loss.item(), global_step=batches)
        
        if cfg.EMA:
            ema_model.update_parameters(model)
            ema_state_extractor.update_parameters(state_extractor)
            ema_time_extractor.update_parameters(time_extractor)

        lr_sched.step()
        batches += 1

    logger.flush()

    # save model
    if len(epoch_losses) < cfg.ITERS_PER_EPOCH/2:
        print("LOSSES ARE MOSTLY NAN")

    final_loss = np.mean(epoch_losses[-20:])
    print(final_loss, float(len(epoch_losses)) / cfg.ITERS_PER_EPOCH)
    if final_loss < best_loss:
        torch.save(model.state_dict(), cfg.OUTPUT_DIR+"/raw_ddpm_model.pt")
        torch.save(state_extractor.state_dict(), cfg.OUTPUT_DIR+"/state_ex_model.pt")
        torch.save(time_extractor.state_dict(), cfg.OUTPUT_DIR+"/tf_model.pt")

        if cfg.EMA:
            torch.save(ema_model.state_dict(), cfg.OUTPUT_DIR+"/ema_ddpm_model.pt")
            torch.save(ema_state_extractor.state_dict(), cfg.OUTPUT_DIR+"/ema_state_ex_model.pt")
            torch.save(ema_time_extractor.state_dict(), cfg.OUTPUT_DIR+"/ema_tf_model.pt")

        best_loss = final_loss
        print("New best model saved.")

    # test on environment periodically
    if (epoch+1)%50 == 0:
        rewards = []
        for i in range(10):
            r, length = episode_with_dp(env, policy, maxlen=cfg.TESTING_MAXLEN, renders=True)
            rewards.append(r)
            print("    Episode Result:", r)
        print("Average Return:", np.mean(rewards))
        logger.add_scalar("reward", np.mean(rewards), global_step=epoch)
