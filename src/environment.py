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
import time

# build a car racing env with scaled and normalized image states
def make_car_racing(size, seed=None):
	env = gym.make("CarRacing-v3", render_mode="rgb_array", 
		lap_complete_percent=0.95, domain_randomize=False, continuous=True)
	env = gym.wrappers.ResizeObservation(env, (size,size))
	env = gym.wrappers.TransformObservation(env, lambda x: (x.astype(np.float32))/255.0, env.observation_space)
	return env


# run an episode with a diffusion policy
def episode_with_dp(env, policy, maxlen=1000, renders=True, skip_initial_frames=0, return_frames=False, seed=None):
	policy.reset()

	if seed is not None:
		s, _ = env.reset(seed=int(seed))
	else:
		s, _ = env.reset()

	frames = []
	all_action_times = []

	done = False
	reward = 0.0
	eplen = 0
	for i in range(skip_initial_frames):
		s, r, term, trunc, info = env.step(env.action_space.sample()*0.0)
		reward += r
		eplen += 1

	while not done:
		st = time.time()
		a = policy.act(s)
		elpased = time.time() - st
		all_action_times.append(elpased)
		delay = 10 - (elpased * 1000)
		delay = int(max(delay, 1))

		# print("    ", a)
		s, r, term, trunc, info = env.step(a)
		if renders:
			ra = env.render()
			cv2.imshow("state", ra[:,:,::-1])
			cv2.waitKey(delay)

			if return_frames:
				frames.append(ra[:,:,::-1])

		eplen += 1
		reward += r
		done = (term or trunc) or eplen>=maxlen

	avg_gen_time = np.mean(policy.gen_time_history)
	print("Average Gen Time:", avg_gen_time, "FPS:", 1.0/avg_gen_time)
	print("Average FPS on all actions:", 1.0/np.mean(all_action_times))

	if return_frames:
		return reward, eplen, frames
	return reward, eplen


# run parallel episodes with a diffusion policy
# a single episode is visualized while others are headless
def parallel_episodes_with_dp(envs, policy, maxlen=1000):
	seeds = np.random.randint(0, high=99999999, size=len(envs))

	policy.reset()
	states = []
	for i in range(len(envs)):
		env = envs[i]
		s, _ = env.reset(seed=int(seeds[i]))
		states.append(s)
	
	need_results = []
	rewards = []
	lengths = []
	for env in envs:
		need_results.append(True)
		rewards.append(0.0)
		lengths.append(0)

	while any(need_results):
		actions = policy.act(states)

		for i in range(len(envs)):
			env = envs[i]
			# print("env", i, "taking action", actions[i])
			s, r, term, trunc, info = env.step(actions[i])
			done = (term or trunc) or lengths[i]>=maxlen

			if need_results[i]:
				rewards[i] += r
				lengths[i] += 1

			if done:
				s, _ = env.reset()
				need_results[i] = False

			states[i] = s

			if i==0:
				ra = env.render()
				cv2.imshow("state", ra[:,:,::-1])
				cv2.waitKey(1)

	return rewards, lengths