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
from .unet_parts_shortcut import ResConvBlockWithStepSize, UNetPairWithStepSize

# U-Net, no attention for now.
class UNetBackbone(torch.nn.Module):
	def __init__(
		self, 
		action_size, # what is the size of each action? (int)
		action_sequence_length, # how many actions are we generating (filter dimension)
		time_condition_dimension, 
		states_condition_dimension, 
		step_size_condition_dimension = None,
		filters = [16, 32, 64, 64],
		pools = [True, True, True],
		channels_are_actions=True,
		**kwargs
		):
		super().__init__()

		if not channels_are_actions:
			action_size, action_sequence_length = action_sequence_length, action_size

		pair_class = UNetPairWithStepSize if (step_size_condition_dimension is not None) else UNetPair
		block_class = ResConvBlockWithStepSize if (step_size_condition_dimension is not None) else ResConvBlock

		base_filters = filters[0]
		self.intro_conv_1 = torch.nn.Conv1d(action_sequence_length, base_filters, 3, 1, padding="same")
		self.pairs = torch.nn.ModuleList()
		for i in range(len(filters)-1):
			# use time_dim as the condition_dim
			mod = pair_class(
				filters[i], 
				filters[i+1], 
				time_condition_dimension, 
				states_condition_dimension, 
				has_attention=False, 
				block_multiplier=1,
				pools=pools[i],
				stepsize_dim=step_size_condition_dimension
			)
			self.pairs.append(mod)

		self.middle_conv = block_class(filters[-1], filters[-1], time_condition_dimension, states_condition_dimension, stepsize_dim=step_size_condition_dimension)
		self.outro_conv = block_class(base_filters, action_sequence_length, time_condition_dimension, states_condition_dimension, stepsize_dim=step_size_condition_dimension)

		self.time_fc1 = torch.nn.Linear(time_condition_dimension, time_condition_dimension)
		self.cond_fc1 = torch.nn.Linear(states_condition_dimension, states_condition_dimension)

		if step_size_condition_dimension is not None:
			self.d_fc1 = torch.nn.Linear(step_size_condition_dimension, step_size_condition_dimension)

	def forward(self, x, t, c, d=None):
		'''
		x (B, seqlen, act_size): float - noised input we are refining into actions
		t (B, time_feats): float - time condition for the diffusion (already as features)
		c (B, state_feats): float - condition for the diffusion, in this case a short state history
		condition c will be a feature vector, possibly pre-processed elsewhere (i.e. if states are images)
		optional (shortcut model): d (B, step_size_feats) - step size ocndition for shortcut models, already as features
		'''
		t_emb = self.time_fc1(t)
		c_emb = self.cond_fc1(c)
		if d is not None:
			d_emb = self.d_fc1(d)
			if len(d_emb.shape)==1:
				# no batch
				d_emb = d_emb[None,:]
		else:
			d_emb = None

		if len(t_emb.shape)==1:
			# no batch
			t_emb = t_emb[None,:]

		if len(c_emb.shape)==1:
			# no batch
			c_emb = c_emb[None,:]

		x = swish(self.intro_conv_1(x))

		for pair in self.pairs:
			x = pair.down(x,t_emb,c_emb,d_emb=d_emb)

		x = self.middle_conv(x,t_emb,c_emb,d_emb=d_emb)

		for pair in self.pairs[::-1]:
			x = pair.up(x,t_emb,c_emb,d_emb=d_emb)

		x = self.outro_conv(x,t_emb,c_emb,d_emb=d_emb)
		return x