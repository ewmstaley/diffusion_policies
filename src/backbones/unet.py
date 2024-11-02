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
UNet Backbone for diffusion. 
'''

import torch
from .unet_parts import swish, PositionalEncoding, AttentionBlock, ResConvBlock, UNetPair

# U-Net, no attention for now.
class UNetBackbone(torch.nn.Module):
	def __init__(
		self, 
		action_size, # what is the size of each action? (int)
		action_sequence_length, # how many actions are we generating (filter dimension)
		time_condition_dimension, 
		states_condition_dimension, 
		filters = [16, 32, 64, 64],
		pools = [True, True, True],
		channels_are_actions=True,
		**kwargs
		):
		super().__init__()

		if not channels_are_actions:
			action_size, action_sequence_length = action_sequence_length, action_size

		base_filters = filters[0]
		self.intro_conv_1 = torch.nn.Conv1d(action_sequence_length, base_filters, 3, 1, padding="same")
		self.pairs = torch.nn.ModuleList()
		for i in range(len(filters)-1):
			# use time_dim as the condition_dim
			mod = UNetPair(
				filters[i], 
				filters[i+1], 
				time_condition_dimension, 
				states_condition_dimension, 
				has_attention=False, 
				block_multiplier=1,
				pools=pools[i]
			)
			self.pairs.append(mod)

		self.middle_conv = ResConvBlock(filters[-1], filters[-1], time_condition_dimension, states_condition_dimension)
		self.outro_conv = ResConvBlock(base_filters, action_sequence_length, time_condition_dimension, states_condition_dimension)

		self.time_fc1 = torch.nn.Linear(time_condition_dimension, time_condition_dimension)
		self.cond_fc1 = torch.nn.Linear(states_condition_dimension, states_condition_dimension)

	def forward(self, x, t, c):
		'''
		x (B, seqlen, act_size): float - noised input we are refining into actions
		t (B, time_feats): float - time condition for the diffusion (already as features)
		c (B, state_feats): float - condition for the diffusion, in this case a short state history
		condition c will be a feature vector, possibly pre-processed elsewhere (i.e. if states are images)
		'''
		t_emb = self.time_fc1(t)
		c_emb = self.cond_fc1(c)

		if len(t_emb.shape)==1:
			# no batch
			t_emb = t_emb[None,:]

		if len(c_emb.shape)==1:
			# no batch
			c_emb = c_emb[None,:]

		x = swish(self.intro_conv_1(x))

		for pair in self.pairs:
			x = pair.down(x,t_emb,c_emb)

		x = self.middle_conv(x,t_emb,c_emb)

		for pair in self.pairs[::-1]:
			x = pair.up(x,t_emb,c_emb)

		x = self.outro_conv(x,t_emb,c_emb)
		return x