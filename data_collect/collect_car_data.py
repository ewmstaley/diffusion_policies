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
Script to collect examples with a joystick. May need to be adjusted for your controller.
'''

import gymnasium as gym
import time
import cv2
import pygame
import numpy as np
import pickle
import random
import string

pygame.init()
pygame.joystick.init()
joystick = pygame.joystick.Joystick(0)
joystick.init()

env = gym.make("CarRacing-v3", render_mode="rgb_array", 
	lap_complete_percent=0.95, domain_randomize=False, continuous=True)

s, _ = env.reset()
done = False

current_action = np.array([0.0, 0.0, 0.0])
axis_states = [0.0, -1.0, -1.0]
rew = 0.0
length = 0
tuples = []

while not done:

	for event in pygame.event.get():
		if event.type == pygame.JOYAXISMOTION:
			# steering 
			if event.axis == 0:
				axis_states[0] = event.value

			# acceleration
			if event.axis == 5:
				axis_states[1] = event.value
				current_action[1] = (event.value+1.0)/2.0
				current_action[1] = np.clip(current_action[1], 0.0, 1.0)

			# break
			if event.axis == 4:
				axis_states[2] = event.value
				current_action[2] += event.value*0.1
				current_action[2] = np.clip(current_action[2], 0.0, 1.0)

	# update the actions based on axis states
	current_action[0] = axis_states[0]
	current_action[0] = np.clip(current_action[0], -1.0, 1.0)
	current_action[1] = (axis_states[1]+1.0)/2.0
	current_action[1] = np.clip(current_action[1], 0.0, 1.0)
	current_action[2] += axis_states[2]*0.1
	current_action[2] = np.clip(current_action[2], 0.0, 1.0)

	# clamp
	thresh = [0.3, 0.1, 0.05]
	if abs(current_action[0]) < thresh[0]:
		current_action[0] = 0.0
	if abs(current_action[1]) < thresh[1]:
		current_action[1] = 0.0
	if abs(current_action[2]) < thresh[2]:
		current_action[2] = 0.0

	# a = env.action_space.sample()
	ra = env.render()
	ns, r, done, trunc, info = env.step(current_action)
	tuples.append({
		"state":np.copy(s), 
		"action":np.copy(current_action), 
		"reward":r, 
		"next_state":np.copy(ns), 
		"done":done
		})
	s = ns
	cv2.imshow("state", ra[:,:,::-1])
	cv2.waitKey(1)
	time.sleep(0.045)

	rew += r
	length += 1
	print(length, rew, current_action)

	# decay steer
	current_action[0] *= 0.95

	should_save = False
	if done and length < 1000:
		folder = "excellent_runs"
		should_save = True
	elif length >= 1000:
		should_save = True
		done = True
		if rew > 800:
			folder = "800_runs"
		elif rew > 700:
			folder = "700_runs"
		else:
			should_save = False

	if should_save:	
		id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
		pickle.dump( (tuples, rew, length), open( "./"+folder+"/run_"+id+".p", "wb" ) )
