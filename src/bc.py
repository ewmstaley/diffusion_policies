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
Script to run and test a simple behavior cloning routine.
'''

import glob
import numpy as np
import torch
import pickle
from buffer import TrajectoryBuffer
from conditional_features import CNNFeatures, SwishModule
from environment import make_car_racing

# ==========================================================
# load in the expert data

HISTORY = 4

buffer = TrajectoryBuffer(
    shapes=[(96,96,3), (3,)], 
    sequence_length=HISTORY
)

# -- car racing data -- #
rews = []
lens = []
for f in glob.glob("../data_collect/excellent_runs/*.p"):
    print(f)
    tuples, reward, length = pickle.load( open( f, "rb" ) )
    rews.append(reward)
    lens.append(length)
    states = [x["state"] for x in tuples]
    actions = [x["action"] for x in tuples]

    # normalize the states
    states = [s.astype(np.float32)/255.0 for s in states]

    # NOTE: Currently do not handle "history" for early states,
    # will add this if it seems needed.

    buffer.add_trajectory([states, actions])

print("Reward:", np.mean(rews), "+-", np.std(rews))
print("Length:", np.mean(lens), "+-", np.std(lens), "Total:", np.sum(lens))

# ==========================================================
# set up a simple CNN model

device = torch.device("cuda")

model = torch.nn.Sequential(
    CNNFeatures(96, 3*HISTORY, layers=4, channels=64, feature_size=128),
    torch.nn.Linear(128, 128),
    SwishModule(),
    torch.nn.Linear(128, 3)
).to(device)

# ==========================================================
# train

ITERS = 5000
opt = torch.optim.Adam(model.parameters(), lr=0.001)
sched = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1.0, end_factor=0.01, total_iters=ITERS)

for batch in range(ITERS):
    opt.zero_grad()
    B = 32
    steps = buffer.sample(B) # [[(B,S), (B,A)], [(B,S), (B,A)], [(B,S), (B,A)], ...]
    history = [x[0] for x in steps] # [(B,S), (B,S), ...]
    targets = steps[-1][1] # (B,A)

    history = np.stack(history, axis=-1) # (B, S, history_seq)
    history = np.reshape(history, (*history.shape[:3], -1)) # (B, w, h, ch*4)
    history = np.transpose(history, (0,3,1,2)) # (B, ch*4, w, h)
    history = torch.from_numpy(history).to(torch.float32).to(device)
    targets = torch.from_numpy(targets).to(torch.float32).to(device)

    a_pred = model(history)
    loss = torch.nn.functional.mse_loss(a_pred, targets)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()

    print(batch, loss.item())

# ==========================================================
# test
env = make_car_racing(96)
rews = []
lens = []
for i in range(100):
    state_history = []
    s, _ = env.reset()
    for h in range(HISTORY):
        state_history.append(s)
    done = False
    reward = 0
    length = 0
    for k in range(1000):
        current_history = state_history[-HISTORY:] #(Hist, W, Height, C)
        current_history = np.stack(current_history, axis=-1) #(W, H, C, hist)
        current_history = np.reshape(current_history, (96, 96, -1))
        current_history = np.transpose(current_history, (2,0,1)) # (C*hist, W, H)
        current_history = torch.from_numpy(current_history).to(torch.float32).to(device)

        a = model(current_history).detach().cpu().numpy()
        a = np.clip(a, np.array([-1, 0, 0]), np.array([1, 1, 1]))
        s, r, term, trunc, info = env.step(a)
        state_history.append(s)
        reward += r
        length += 1
        done = (term or trunc)

    print(reward)
    rews.append(reward)
    lens.append(length)

print("Reward:", np.mean(rews), "+-", np.std(rews))
print("Length:", np.mean(lens), "+-", np.std(lens), "Total:", np.sum(lens))
