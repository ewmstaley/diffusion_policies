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
Utilities for building and interacting with the car racing environment.
'''

import gymnasium as gym
import numpy as np
import cv2

# build a car racing env with scaled and normalized image states
def make_car_racing(size):
	env = gym.make("CarRacing-v3", render_mode="rgb_array", 
		lap_complete_percent=0.95, domain_randomize=False, continuous=True)
	env = gym.wrappers.ResizeObservation(env, (size,size))
	env = gym.wrappers.TransformObservation(env, lambda x: (x.astype(np.float32))/255.0, env.observation_space)
	return env

# run an episode with a diffusion policy
def episode_with_dp(env, policy, maxlen=1000, renders=True, skip_initial_frames=0, return_frames=False):
	policy.reset()
	s, _ = env.reset()

	frames = []

	done = False
	reward = 0.0
	eplen = 0
	for i in range(skip_initial_frames):
		s, r, term, trunc, info = env.step(env.action_space.sample()*0.0)
		reward += r
		eplen += 1

	while not done:
		a = policy.act(s)
		# print("    ", a)
		s, r, term, trunc, info = env.step(a)
		if renders:
			ra = env.render()
			cv2.imshow("state", ra[:,:,::-1])
			cv2.waitKey(1)

			if return_frames:
				frames.append(ra[:,:,::-1])

		eplen += 1
		reward += r
		done = (term or trunc) or eplen>=maxlen

	if return_frames:
		return reward, eplen, frames
	return reward, eplen