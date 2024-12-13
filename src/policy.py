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
import time
from utility import get_diffusion_parameters
from tqdm import tqdm
import copy


class DiffusionPolicy():

	def __init__(self, state_shape, action_size, model, state_extractor, time_extractor, 
		device, sampling_algo, step_size_extractor=None,
		T=1000, history_size=4, action_sequence_length=16, channels_are_actions=True, clip=None, parallel_envs=False):

		self.state_shape = state_shape
		self.action_size = action_size
		self.model = model
		self.state_extractor = state_extractor
		self.time_extractor = time_extractor
		self.step_size_extractor = step_size_extractor
		self.device = device

		self.state_history = []
		self.current_trajectory = []
		self.gen_time_history = [] # to record how long it takes, on average, to generate

		self.T = T
		self.history_size = history_size
		self.action_sequence_length = action_sequence_length
		self.sampling_algo = sampling_algo
		self.channels_are_actions = channels_are_actions
		self.clip = clip

		# are we actually dealing with batches of states?
		self.parallel_envs = parallel_envs


	def reset(self):
		self.state_history = []
		self.current_trajectory = []
		self.gen_time_history = []


	def act(self, state):
		if not self.parallel_envs:
			# process a "set of one environments" in parallel
			state = [state]

		self.state_history.append(copy.deepcopy(state))

		if len(self.current_trajectory)==0:

			# make sure history is at least self.history_size
			while len(self.state_history) < self.history_size:
				self.state_history.append(self.state_history[-1])

			# make sure it is no more than self.history_size
			if len(self.state_history) > self.history_size:
				self.state_history = self.state_history[-self.history_size:]

			# make a new trajectory
			stx = time.time()
			self.refresh_trajectory()
			elapsedx = time.time() - stx
			self.gen_time_history.append(elapsedx)
		
		# note: current_trajectory is shape (envs, action_horizon_over_2, action_dim)
		action = self.current_trajectory[:,0,:]
		if not self.parallel_envs:
			action = action[0]
		self.current_trajectory = self.current_trajectory[:,1:,:]

		# if second dim is zero, we have exhausted our actions
		if self.current_trajectory.shape[1] == 0:
			self.current_trajectory = []
		return action


	def refresh_trajectory(self):
		# eval mode
		self.model.eval()
		self.state_extractor.eval()
		self.time_extractor.eval()

		# get state history as a tensor
		history = np.stack(self.state_history, axis=-1) # (envs, W, H, ch, hist) or (envs, S, hist)
		E = history.shape[0]
		if len(self.state_shape)==3:
			W = history.shape[1]
			history = np.reshape(history, (E, W, W, -1))
			history = np.transpose(history, (0,3,1,2)) # (E, ch*hist, W, H)
		else:
			history = np.reshape(history, (E, -1))
		history = torch.from_numpy(history).to(torch.float32).to(self.device)

		# get times
		ts = self.sampling_algo.ts
		if self.sampling_algo.reverse_gen:
			ts = ts[::-1] 
		ts = torch.as_tensor(ts).to(torch.long).to(self.device) # (B, t)
		sampling_d = self.sampling_algo.sampling_d

		# get state and condition vector - only need to do this once
		state_features = self.state_extractor(history)
		if len(state_features.shape) == 1:
			state_features = state_features[None, :]

		if sampling_d is not None and self.step_size_extractor is not None:
			step_size_features = self.step_size_extractor(sampling_d)
		else:
			step_size_features = None

		# generate ---------------------------------------------------------------
		# start with random noise, x
		if self.channels_are_actions:
			x = torch.randn((E,self.action_sequence_length,self.action_size))
		else:
			x = torch.randn((E,self.action_size,self.action_sequence_length))
		x = x.to(torch.float32).to(self.device)

		with torch.no_grad():
			for t in ts:

				time_features = self.time_extractor(t)
				if len(time_features.shape) == 1:
					time_features = time_features[None, :]
				if E>1:
					time_features = time_features.repeat(E,1)

				# predict the noise
				pred_noise = self.model(x, time_features, state_features, d=step_size_features)

				# backwards diffusion
				x = self.sampling_algo.generation(x, t, pred_noise)

		# convert output to numpy
		x = x.data.cpu().numpy()
		if not self.channels_are_actions:
			x = np.transpose(x, (0,2,1))

		if self.clip is not None:
			x = np.clip(x, self.clip[0], self.clip[1])

		x = x[:,:self.action_sequence_length//2]

		self.current_trajectory = x

		# flip back to train mode
		self.model.train()
		self.state_extractor.train()
		self.time_extractor.train()