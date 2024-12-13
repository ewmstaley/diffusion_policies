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

import torch
import numpy as np
from backbones.mlp import MLPBackbone
from backbones.unet import UNetBackbone
import cv2
import math
from tqdm import tqdm
from utility import get_diffusion_parameters, cosine_lr_scheduler, sample_ts_by_ds
import matplotlib.pyplot as plt
import os
import pickle
from box import Box
import yaml
import glob
from buffer import TrajectoryBuffer
from conditional_features import StateVectorFeatures, CNNFeatures, TimeFeatures, StepSizeFeatures
from itertools import chain
from torch.utils.tensorboard import SummaryWriter
from policy import DiffusionPolicy
from environment import make_car_racing, episode_with_dp
from algos import DDPMAlgo, DDIMAlgo, FlowMatchingAlgo, FlowMatchingQuadraticAlgo

cfg = Box()
# =====================================================
cfg.BACKBONE = "UNET"
cfg.ALGO = "FLOW" # use FLOW for shortcut models
cfg.SHORTCUT = True # always True for this script
cfg.STEP_SIZE = 32
cfg.ACTION_SIZE = 3
cfg.STATE_SHAPE = (96,96,3)
cfg.STATE_HISTORY = 4
cfg.NUM_ACTIONS = 8
cfg.CHANNELS_ARE_ACTIONS = False
cfg.T = 128
cfg.BATCH_SIZE_NORMAL = 24 # batch size for standard flow objective
cfg.BATCH_SIZE_CONSISTENCY = 8 # batch size for self-consistency
cfg.EPOCHS = 1000
cfg.ITERS_PER_EPOCH = 500
cfg.EMA = True
cfg.OUTPUT_DIR = "./output_flow_shortcut/"
cfg.TESTING_MAXLEN = 1000
cfg.K = 256 # used for both embedding sizes and number of channels
# =====================================================

if not os.path.exists(cfg.OUTPUT_DIR):
    os.makedirs(cfg.OUTPUT_DIR)

# save config
with open(cfg.OUTPUT_DIR+'config.yml', 'w') as outfile:
    d = cfg.to_dict()
    d["STATE_SHAPE"] = list(d["STATE_SHAPE"]) # avoid annoying yaml error with tuples
    yaml.dump(d, outfile)

# get data loaded in nice format
buffer = TrajectoryBuffer(
    shapes=[cfg.STATE_SHAPE, (cfg.ACTION_SIZE,)], 
    sequence_length=cfg.STATE_HISTORY+cfg.NUM_ACTIONS-1
)

# -- car racing -- #
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

if cfg.ALGO == "DDPM":
    algo = DDPMAlgo(cfg.T)
elif cfg.ALGO == "DDIM":
    algo = DDIMAlgo(cfg.T, cfg.STEP_SIZE)
elif cfg.ALGO == "FLOW":
    algo = FlowMatchingAlgo(cfg.T, cfg.STEP_SIZE)
elif cfg.ALGO == "FLOWQ":
    algo = FlowMatchingQuadraticAlgo(cfg.T, cfg.STEP_SIZE)
else:
    raise ValueError("cfg.ALGO must be one of [DDPM, DDIM, FLOW]")

# =====================================================

device = torch.device("cuda")

# set up feature extractors
if len(cfg.STATE_SHAPE) == 3:
    state_extractor = CNNFeatures(initial_side=cfg.STATE_SHAPE[0], initial_channels=cfg.STATE_SHAPE[-1]*cfg.STATE_HISTORY, 
        layers=4, channels=48, feature_size=cfg.K).to(device)
else:
    state_extractor = StateVectorFeatures(initial_size=cfg.STATE_SHAPE[0]*cfg.STATE_HISTORY, feature_size=cfg.K).to(device)

time_extractor = TimeFeatures(times=cfg.T, feature_size=cfg.K).to(device)
step_size_extractor = StepSizeFeatures(options=8, feature_size=cfg.K).to(device)

# currently only support UNET, but this is a good spot to add other backbones later
if cfg.BACKBONE == "MLP":
    pass
elif cfg.BACKBONE == "UNET":
    model = UNetBackbone(
        action_size = cfg.ACTION_SIZE, 
        action_sequence_length = cfg.NUM_ACTIONS, 
        time_condition_dimension = cfg.K, 
        states_condition_dimension = cfg.K, 
        step_size_condition_dimension = cfg.K,
        filters = [cfg.K, cfg.K, cfg.K, cfg.K],
        pools = [True, True, False],
        channels_are_actions = cfg.CHANNELS_ARE_ACTIONS
    ).to(device)


all_params = chain(state_extractor.parameters(), time_extractor.parameters(), 
    step_size_extractor.parameters(), model.parameters())

opt = torch.optim.Adam(all_params, lr=0.0001, weight_decay=1e-4)
total_steps = cfg.ITERS_PER_EPOCH*cfg.EPOCHS
lr_sched = cosine_lr_scheduler(opt, total_steps, warmup_steps=500, final=0.001)

if cfg.EMA:
    ema_model = torch.optim.swa_utils.AveragedModel(model, 
        multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999))
    ema_state_extractor = torch.optim.swa_utils.AveragedModel(state_extractor, 
        multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999))
    ema_time_extractor = torch.optim.swa_utils.AveragedModel(time_extractor, 
        multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999))
    ema_step_extractor = torch.optim.swa_utils.AveragedModel(step_size_extractor, 
        multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999))

state_ex_params = sum(p.numel() for p in state_extractor.parameters())
tf_params = sum(p.numel() for p in time_extractor.parameters())
df_params = sum(p.numel() for p in step_size_extractor.parameters())
backbone_params = sum(p.numel() for p in model.parameters())
print("---")
print("State Extractor has", f'{state_ex_params:,}', "parameters.")
print("Time Extractor has", f'{tf_params:,}', "parameters.")
print("Step Size Extractor has", f'{df_params:,}', "parameters.")
print("Backbone has", f'{backbone_params:,}', "parameters.")
print("---")

# =====================================================

logger = SummaryWriter(log_dir=cfg.OUTPUT_DIR)

# train
best_loss = 100000.0
loss_history = []
batches = 0

for epoch in range(cfg.EPOCHS):
    print("EPOCH", epoch+1)

    epoch_losses = []
    epoch_loss1_history = []
    epoch_loss2_history = []
    for b in tqdm(range(cfg.ITERS_PER_EPOCH)):
        opt.zero_grad()
        B = cfg.BATCH_SIZE_NORMAL + cfg.BATCH_SIZE_CONSISTENCY

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

        history1 = history[:cfg.BATCH_SIZE_NORMAL]
        history2 = history[cfg.BATCH_SIZE_NORMAL:]

        targets = np.stack(targets, axis=-1) # (B, A, seq)
        if cfg.CHANNELS_ARE_ACTIONS:
            targets = np.transpose(targets, (0,2,1)) # (B, seq, A)
        targets = torch.from_numpy(targets).to(torch.float32).to(device)

        targets1 = targets[:cfg.BATCH_SIZE_NORMAL]
        targets2 = targets[cfg.BATCH_SIZE_NORMAL:]

        loss = 0

        # FIRST BATCH: normal diffusion or flow matching objective
        # =========================================================================
        targets = targets1
        history = history1
        e = torch.randn(*targets.shape).to(torch.float32).to(device)
        t = np.random.randint(0, cfg.T, size=(cfg.BATCH_SIZE_NORMAL,))
        t = torch.as_tensor(t).to(torch.long).to(device)
        d = torch.zeros_like(t)

        state_features = state_extractor(history)
        time_features = time_extractor(t)
        step_size_features = step_size_extractor(d)

        noised_batch, training_target = algo.forward_diffusion(targets, e, t)
        predicted_noise = model(noised_batch, time_features, state_features, step_size_features)
        loss1 = torch.nn.functional.mse_loss(predicted_noise, training_target)
        loss += loss1

        # SECOND BATCH: self-consistency
        # =========================================================================
        targets = targets2
        history = history2
        e = torch.randn(*targets.shape).to(torch.float32).to(device)
        d = np.random.randint(0, 7, size=(cfg.BATCH_SIZE_CONSISTENCY,))

        # constrain t's to times we could hit during inference under the above d's
        t = sample_ts_by_ds(d, cfg.T)

        # convert
        d = torch.as_tensor(d).to(torch.long).to(device)
        t = torch.as_tensor(t).to(torch.long).to(device)
        step_sizes_steps = torch.pow(torch.ones_like(d)*2, d)
        step_sizes_fractional = step_sizes_steps / float(cfg.T)

        B, d1, d2 = e.shape
        step_sizes_fractional = step_sizes_fractional[:,None,None].repeat(1, d1, d2)

        # prepare targets
        state_features = state_extractor(history)
        time_features = time_extractor(t)
        xt, _ = algo.forward_diffusion(targets, e, t)

        # TODO: Should probably move to target.detach() instead of no_grad()
        # I am unclear if no_grad does something here since the loss computation is outside it.
        with torch.no_grad():
            t_feats_ema = ema_time_extractor(t)
            s_feats_ema = ema_state_extractor(history)
            ss_feats_ema = ema_step_extractor(d)

            # take small step
            st = model(xt, t_feats_ema, s_feats_ema, ss_feats_ema)
            xt_plus_d = xt + st*step_sizes_fractional

            # take step again
            time_features_next = ema_time_extractor(t+step_sizes_steps)
            st_plus_d = model(xt_plus_d, time_features_next, s_feats_ema, ss_feats_ema)

            batch_2_target_velocities = (st + st_plus_d)*0.5

        d = d+1
        step_size_features = step_size_extractor(d)
        pred = model(xt, time_features, state_features, step_size_features)
        loss2 = torch.nn.functional.mse_loss(pred, batch_2_target_velocities)
        loss += loss2

        # =========================================================================

        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        opt.step()

        if not math.isnan(loss.item()):
            epoch_loss1_history.append(loss1.item())
            epoch_loss2_history.append(loss2.item())
            epoch_losses.append(loss.item())
            logger.add_scalar("loss", loss.item(), global_step=batches)
        
        if cfg.EMA:
            ema_model.update_parameters(model)
            ema_state_extractor.update_parameters(state_extractor)
            ema_time_extractor.update_parameters(time_extractor)
            ema_step_extractor.update_parameters(step_size_extractor)

        lr_sched.step()
        batches += 1

    logger.flush()

    # save model
    if len(epoch_losses) < cfg.ITERS_PER_EPOCH/2:
        print("LOSSES ARE MOSTLY NAN")

    final_loss = np.mean(epoch_losses[-20:])
    print(final_loss, float(len(epoch_losses)) / cfg.ITERS_PER_EPOCH)
    print(np.mean(epoch_loss1_history[-20:]), np.mean(epoch_loss2_history[-20:]))
    if final_loss < best_loss:
        torch.save(model.state_dict(), cfg.OUTPUT_DIR+"/raw_ddpm_model.pt")
        torch.save(state_extractor.state_dict(), cfg.OUTPUT_DIR+"/state_ex_model.pt")
        torch.save(time_extractor.state_dict(), cfg.OUTPUT_DIR+"/tf_model.pt")
        torch.save(step_size_extractor.state_dict(), cfg.OUTPUT_DIR+"/stepf_model.pt")

        if cfg.EMA:
            torch.save(ema_model.state_dict(), cfg.OUTPUT_DIR+"/ema_ddpm_model.pt")
            torch.save(ema_state_extractor.state_dict(), cfg.OUTPUT_DIR+"/ema_state_ex_model.pt")
            torch.save(ema_time_extractor.state_dict(), cfg.OUTPUT_DIR+"/ema_tf_model.pt")
            torch.save(ema_step_extractor.state_dict(), cfg.OUTPUT_DIR+"/ema_stepf_model.pt")

        best_loss = final_loss
        print("New best model saved.")