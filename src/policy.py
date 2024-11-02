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
Class to wrap the diffusion model and execute it as a policy.
Keeps track of things like when to generate actions, state history, etc.
'''

import torch
import numpy as np
import cv2
import math
from utility import get_diffusion_parameters
from tqdm import tqdm


class DiffusionPolicy():

	def __init__(self, state_shape, action_size, model, state_extractor, time_extractor, device, T=1000,
		history_size=4, action_sequence_length=16, implicit=True, sampling_factor=20, channels_are_actions=True,
		clip=None):

		self.state_shape = state_shape
		self.action_size = action_size
		self.model = model
		self.state_extractor = state_extractor
		self.time_extractor = time_extractor
		self.device = device

		self.state_history = []
		self.current_trajectory = []

		self.T = T
		self.history_size = history_size
		self.action_sequence_length = action_sequence_length
		self.implicit = implicit
		self.time_sampling_factor = sampling_factor
		self.channels_are_actions = channels_are_actions
		self.clip = None


	def reset(self):
		self.state_history = []
		self.current_trajectory = []


	def act(self, state):
		self.state_history.append(state)

		if len(self.current_trajectory)==0:

			# make sure history is at least self.history_size
			while len(self.state_history) < self.history_size:
				self.state_history.append(self.state_history[-1])

			# make sure it is no more than self.history_size
			if len(self.state_history) > self.history_size:
				self.state_history = self.state_history[-self.history_size:]

			# make a new trajectory
			self.refresh_trajectory()

		action = self.current_trajectory[0]
		self.current_trajectory = self.current_trajectory[1:]
		return action


	def refresh_trajectory(self):
		# eval mode
		self.model.eval()
		self.state_extractor.eval()
		self.time_extractor.eval()

		# get state history as a tensor
		history = np.stack(self.state_history, axis=-1) # (W, H, ch, hist) or (S, hist)
		if len(self.state_shape)==3:
			W = history.shape[0]
			history = np.reshape(history, (W, W, -1))
			history = np.transpose(history, (2,0,1)) # (ch*hist, W, H)
		else:
			history = history.flatten()
		history = torch.from_numpy(history).to(torch.float32).to(self.device)

		# get times
		ts = list(range(self.T))
		ts = ts[::self.time_sampling_factor][::-1] 
		ts = torch.as_tensor(ts).to(torch.long).to(self.device) # (B, t)

		# get state and condition vector - only need to do this once
		state_features = self.state_extractor(history)
		if len(state_features.shape) == 1:
			state_features = state_features[None, :]

		# get diffusion params
		betas, alphas, alpha_hats = get_diffusion_parameters(self.T)

		# generate ---------------------------------------------------------------
		# start with random noise, x
		if self.channels_are_actions:
			x = torch.randn((1,self.action_sequence_length,self.action_size))
		else:
			x = torch.randn((1,self.action_size,self.action_sequence_length))
		x = x.to(torch.float32).to(self.device)

		with torch.no_grad():
			for t in ts:

				time_features = self.time_extractor(t)
				if len(time_features.shape) == 1:
					time_features = time_features[None, :]

				# added noise (only for non-implicit)
				z = torch.randn(*x.shape).to(torch.float32).to(self.device)
				if t == 0:
					z *= 0

				# get params for this step
				beta = betas[t]
				alpha = alphas[t]
				alphahat = alpha_hats[t]

				# predict the noise
				pred_noise = self.model(x, time_features, state_features)

				if self.implicit:
					# DDIM uses alphaHAT but calls it alpha
					# I am redefining alpha to be alpha_hat here
					alpha = alphahat
					alphas = alpha_hats

					atm1 = alphas[t-self.time_sampling_factor] if t>0 else 1.0

					pred_x0 = (x - math.sqrt(1.0-alpha)*pred_noise)/math.sqrt(alpha)
					point_xt = math.sqrt(1.0 - atm1)*pred_noise

					x = math.sqrt(atm1)*pred_x0 + point_xt
				else:
					x = (1.0/math.sqrt(alpha))*(x - ((1.0 - alpha)/math.sqrt(1.0 - alphahat))*pred_noise) + math.sqrt(beta)*z

		# convert output to numpy
		x = x.data.cpu().numpy()
		if not self.channels_are_actions:
			x = np.transpose(x, (0,2,1))

		if self.clip is not None:
			x = np.clip(x, self.clip[0], self.clip[1])

		x = x[0]
		x = x[:self.action_sequence_length//2]

		self.current_trajectory = x

		# flip back to train mode
		self.model.train()
		self.state_extractor.train()
		self.time_extractor.train()