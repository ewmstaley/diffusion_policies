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
MLP Backbone for diffusion. 
'''

import torch

def swish(x):
	return x*torch.sigmoid(x)

class MLPBackbone(torch.nn.Module):

	def __init__(
		self, 
		action_size, 
		action_sequence_length, 
		time_condition_dimension, 
		states_condition_dimension, 
		**kwargs
		):
		super().__init__()
		W = 2048

		conds = time_condition_dimension+states_condition_dimension
		self.fc1 = torch.nn.Linear(action_size*action_sequence_length, W)
		self.fc2 = torch.nn.Linear(W+conds, W)
		self.fc3 = torch.nn.Linear(W+conds, W)
		self.fc4 = torch.nn.Linear(W+conds, action_size*action_sequence_length)

	def forward(self, x, t, c):
		'''
		x (B, seqlen, act_size): float - noised input we are refining into actions
		t (B, time_feats): float - time condition for the diffusion (already as features)
		c (B, state_feats): float - condition for the diffusion, in this case a short state history
		condition c will be a feature vector, possibly pre-processed elsewhere (i.e. if states are images)
		'''
		
		B, S, A = x.shape
		original = x
		x = x.reshape(B, -1)
		conds = torch.cat([t, c], dim=-1)

		x = swish(self.fc1(x))
		xc = torch.cat([x, conds], dim=-1)
		x = swish(self.fc2(xc))# + x
		xc = torch.cat([x, conds], dim=-1)
		x = swish(self.fc3(xc))# + x
		xc = torch.cat([x, conds], dim=-1)
		x = swish(self.fc4(xc))

		x = x.reshape(B, S, A) + original
		return x

