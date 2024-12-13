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
Unique aspects of each algorithm.
See DDPMAlgo class for notes.

Flow Matching still implements "forward_diffusion" to provide
training targets- the name is just for compatability.
'''

import torch
import numpy as np
import math

class DDPMAlgo():

    def __init__(self, T, **kwargs):
        self.T = T
        self.betas = np.linspace(0.0001, 0.02, T)
        self.alphas = 1.0 - self.betas
        self.alpha_hats = np.cumprod(self.alphas)
        self.ts = list(range(self.T))
        self.sampling_d = None
        self.reverse_gen = True

    def forward_diffusion(self, data, noise, t):
        '''
        Given the unnoised data, a noise tensor of same size, and an integer index t
        returns: (the noised data point, and the MSE training target)
        '''
        ahat = self.alpha_hats[t.detach().cpu().numpy()]
        ahat = torch.as_tensor(ahat).to(torch.float32).to(data.device)
        B, d1, d2 = data.shape
        ahat = ahat.unsqueeze(1).unsqueeze(2).repeat(1,d1,d2)
        noised_batch = torch.sqrt(ahat)*data + torch.sqrt(1.0 - ahat)*noise
        return noised_batch, noise

    def generation(self, x, t, pred_noise):
        '''
        Given a partially refined datapoint x, the time t, and a prediction from our model,
        update and return the new x.
        '''
        z = torch.randn(*x.shape).to(torch.float32).to(self.device)
        if t == 0:
            z *= 0
        beta = self.betas[t]
        alpha = self.alphas[t]
        alphahat = self.alpha_hats[t]
        x = (1.0/math.sqrt(alpha))*(x - ((1.0 - alpha)/math.sqrt(1.0 - alphahat))*pred_noise) + math.sqrt(beta)*z
        return x


class DDIMAlgo(DDPMAlgo):
    def __init__(self, T, step_size=20, **kwargs):
        super().__init__(T=T)
        self.step_size = step_size
        self.ts = list(range(self.T))[::step_size]
        self.reverse_gen = True

    def generation(self, x, t, pred_noise):
        beta = self.betas[t]
        alpha = self.alpha_hats[t] # note this differs from DDPM's alpha
        atm1 = self.alpha_hats[t-self.step_size] if t>0 else 1.0

        pred_x0 = (x - math.sqrt(1.0-alpha)*pred_noise)/math.sqrt(alpha)
        point_xt = math.sqrt(1.0 - atm1)*pred_noise

        x = math.sqrt(atm1)*pred_x0 + point_xt
        return x


class FlowMatchingAlgo():
    def __init__(self, T, step_size=20, **kwargs):
        self.T = T
        self.step_size = step_size
        self.ts = list(range(self.T))[::step_size]
        self.sampling_d = int(np.log2(step_size))
        self.reverse_gen = False

    def forward_diffusion(self, data, noise, t):
        t_fractional = t.to(torch.float32) / float(self.T)
        _, d1, d2 = data.shape
        t_fractional = t_fractional[:,None,None].repeat(1,d1,d2)
        noised_batch = (1.0-t_fractional)*noise + t_fractional*data
        velocity = data - noise
        return noised_batch, velocity

    def generation(self, x, t, pred_noise):
        # here the "predicted noise" is more accurately "velocity"
        # we need to multiply by a scalar to get the appropriate change to x.
        velocity = pred_noise
        step_size_fractional = float(self.step_size)/float(self.T)
        x = x + velocity*step_size_fractional
        return x


# Not used anywhere- an example of a custom flow match trajectory
class FlowMatchingQuadraticAlgo():
    def __init__(self, T, step_size=20, **kwargs):
        self.T = T
        self.step_size = step_size
        self.ts = list(range(self.T))[::step_size]
        self.sampling_d = None
        self.reverse_gen = False

    def forward_diffusion(self, data, noise, t):
        t_fractional = t.to(torch.float32) / float(self.T)
        _, d1, d2 = data.shape
        t_fractional = t_fractional[:,None,None].repeat(1,d1,d2)
        noised_batch = (1.0-t_fractional)*noise + t_fractional*data
        velocity = data - noise
        vel_mult = (t_fractional*2.0) - 2.0 #dy/dt of (t-1)^2
        return noised_batch, velocity*vel_mult

    def generation(self, x, t, pred_noise):
        # here the "predicted noise" is more accurately "velocity"
        # we need to multiply by a scalar to get the appropriate change to x.
        t_fractional = t.to(torch.float32) / float(self.T)
        velocity = pred_noise
        vel_mult = (t_fractional*2.0) - 2.0
        step_size_fractional = float(self.step_size)/float(self.T)
        x = x + velocity*vel_mult*step_size_fractional
        return x
