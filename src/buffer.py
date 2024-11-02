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
Class to store trajectories and sample segments of them.
'''

import numpy as np

class TrajectoryBuffer():

	def __init__(self, shapes, sequence_length):
		self.collections = []
		for s in shapes:
			# Do NOT create empty arrays and then append to them. This created horrible numerical errors.
			# Instead, just instantiate the arrays with the first data we see. We use a None placeholder here.
			self.collections.append(None)
		self.samplable_indices = []
		self.sequence_length = sequence_length

	def add_trajectory(self, data_specific_chains):
		# add trajectory by supplying a list of the sequences for each type
		# for example: [[50 states], [50 actions]] for a trajectory of 50 (s,a) pairs
		# assumes all chains have same length

		start_idx = 0 if self.collections[0] is None else len(self.collections[0])
		ref_len = len(data_specific_chains[0])
		for i,chain in enumerate(data_specific_chains):
			assert len(chain) == ref_len
			if self.collections[i] is None:
				self.collections[i] = np.array(chain, dtype=np.float32)
			else:
				self.collections[i] = np.append(self.collections[i], np.array(chain, dtype=np.float32), axis=0)

		for i in range(ref_len - self.sequence_length + 1):
			self.samplable_indices.append(i + start_idx)

	def sample(self, amount):
		'''
		returns each step of N trajectories. For example, sampling 2 (s,a) trajectories of length 4:
		would return: [
			[[s,s],[a,a]], # all the first states and first actions
			[[s,s],[a,a]], # all the second states and second actions
			[[s,s],[a,a]], # etc
			[[s,s],[a,a]],
		]
		'''
		indices = np.random.choice(self.samplable_indices, size=amount, replace=True)
		indices = np.array(indices)
		steps = []
		for i in range(self.sequence_length):
			steps.append([c[indices] for c in self.collections])
			indices += 1

		return steps