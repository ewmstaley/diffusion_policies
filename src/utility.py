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
Some helper methods.
'''

import torch
import numpy as np
import math

def get_diffusion_parameters(T=1000):
    betas = np.linspace(0.0001, 0.02, T)
    alphas = 1.0 - betas
    alpha_hats = np.cumprod(alphas)
    return betas, alphas, alpha_hats


# not sure we really need something this complex, but it works well...
def cosine_lr_scheduler(opt, total_steps, warmup_steps, final=0.001):

    def thunk(stepnum):
        if stepnum <= warmup_steps:
            # go from ~0 to 1.0
            prog = float(stepnum)/float(warmup_steps)
            lrmult = 0.00001 + prog
        else:
            # go from 1.0 to ~0
            steps_after_peak = stepnum-warmup_steps
            tail_steps = total_steps-warmup_steps
            prog = float(steps_after_peak) / float(tail_steps)
            lrmult = ((np.cos(3.141592*prog)+1.0)*0.5)*(1.0-final) + final
        return max(lrmult, 0.0)

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=thunk)

    return scheduler


# given a setting for d in a shortcut model,
# only sample ts that can be reached under this setting
def sample_ts_by_ds(ds, tmax=128):
    ts = []
    for d in ds:
        stepsize = np.pow(2, d)
        options = np.arange(0, tmax, stepsize)[:-1] 
        # NOTE: we take off the last one since we need to be able to take two steps

        t = np.random.choice(options)
        ts.append(t)
    return np.array(ts)

