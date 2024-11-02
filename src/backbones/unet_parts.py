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
Components to make building a UNet easier.
'''

import torch

def swish(x):
	return x*torch.nn.functional.sigmoid(x)

# stolen and then edited from: https://pytorch.org/tutorials/beginner/translation_transformer.html
# TODO: I think this is duplicated in the conditional features file...
class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, embeddings=1000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, d_model, 2)* math.log(10000) / d_model)
        pos = torch.arange(0, embeddings).reshape(embeddings, 1)
        pos_embedding = torch.zeros((embeddings, d_model))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        # pos_embedding = pos_embedding.unsqueeze(-2)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, t):
    	return self.pos_embedding[t]


# block that computes queries, keys, and values for a sequence of vectors and the applies MHA
# currently unused
class AttentionBlock(torch.nn.Module):
	def __init__(self, in_features):
		super().__init__()
		self.to_qkv = torch.nn.Linear(in_features, in_features*3)
		self.mha = torch.nn.MultiheadAttention(embed_dim=in_features, num_heads=4, batch_first=True)

	def forward(self, x):
		B, in_ch, seq = x.shape
		x = torch.transpose(x, 1, 2) # B, seq, in_ch
		qkv = self.to_qkv(x)
		x, _ = self.mha(qkv[:,:,:in_ch], qkv[:,:,in_ch:2*in_ch], qkv[:,:,-in_ch:], need_weights=False)
		x = torch.transpose(x, 1, 2) # B, in_ch, seq
		return x

'''
Residual convolution block with time and state conditioning.
Takes as input x:(B,in_ch,size), time embedding: (B, t), state condition: (B, s)
Outputs (B, out_ch, size).
'''
class ResConvBlock(torch.nn.Module):
	def __init__(self, in_channels, out_channels, time_dim, condition_dim, has_attention=False):
		super().__init__()
		self.norm1 = torch.nn.GroupNorm(in_channels//4, in_channels)
		self.conv1 = torch.nn.Conv1d(in_channels, out_channels, 3, 1, padding="same")
		self.norm2 = torch.nn.GroupNorm(max(out_channels//4,1), out_channels*2)
		self.conv2 = torch.nn.Conv1d(out_channels*2, out_channels, 3, 1, padding="same")
		self.norm3 = torch.nn.GroupNorm(max(out_channels//4,1), out_channels)
		self.conv3 = torch.nn.Conv1d(out_channels, out_channels, 3, 1, padding="same")
		self.dropout = torch.nn.Dropout(0.1)

		if in_channels!=out_channels:
			self.res_connection = torch.nn.Conv1d(in_channels, out_channels, 1, 1, padding="same")
		else:
			self.res_connection = torch.nn.Identity()

		self.has_attention = has_attention
		if self.has_attention:
			self.norm4 = torch.nn.GroupNorm(max(out_channels//4,1), out_channels)
			self.attn = AttentionBlock(out_channels)

		self.time_scale = torch.nn.Linear(time_dim, out_channels)
		self.time_shift = torch.nn.Linear(time_dim, out_channels)

		self.cond_layer = torch.nn.Linear(condition_dim, out_channels)

	def forward(self, x, t_emb, cond):
		# first conv
		y = swish(self.norm1(x))
		y = self.conv1(y)
		B, ch, sz = y.shape

		# inject condition
		c = swish(cond)
		c = self.cond_layer(c)[:,:,None]
		c = c.repeat(1, 1, sz)
		y = torch.cat((y,c), dim=1) # concat channels

		# second conv
		y = swish(self.norm2(y))
		y = self.conv2(y) # 2x[out_ch] -> out_ch

		# inject time as a scale + offset to our data
		t = swish(t_emb)
		scale = self.time_scale(t)[:,:,None] + 1.0 # the "default" is scale of 1.0
		shift = self.time_shift(t)[:,:,None]
		y = y*scale + shift

		# third conv
		y = swish(self.norm3(y))
		y = self.dropout(y)
		y = self.conv3(y)

		y = y + self.res_connection(x)

		# attention
		if self.has_attention:
			y = self.attn(self.norm4(y)) + y

		return y

'''
A down-sizing block and a corresponding up-sizing block.
Helps to preserve sanity when making the UNet.

The down-sizing block applies a convolutional block and then pools (downsizes).
The up-sizing block upsamples first and then applies a convolutional block.

The content before pooling is saved and concatenated to the data post-upsampling, maintaining the UNet connections
for us if we use several of these blocks.
'''
class UNetPair(torch.nn.Module):
	def __init__(self, in_channels, out_channels, time_dim, condition_dim, has_attention=False, block_multiplier=1, pools=True):
		super().__init__()
		self.pools = pools
		self.dconvs = torch.nn.ModuleList()
		for i in range(block_multiplier):
			in_ch = in_channels if i==0 else out_channels
			self.dconvs.append(ResConvBlock(in_ch, out_channels, time_dim, condition_dim, has_attention=has_attention))
		if self.pools:
			self.pool = torch.nn.MaxPool1d(2)

		self.uconvs = torch.nn.ModuleList()
		for i in range(block_multiplier):
			in_ch = out_channels*2 if i==0 else in_channels
			self.uconvs.append(ResConvBlock(in_ch, in_channels, time_dim, condition_dim, has_attention=has_attention))
		if self.pools:
			self.upsampler = torch.nn.Upsample(scale_factor=2)

		self.connection = None

	def down(self, x, t_emb, cond):
		for dconv in self.dconvs:
			x = dconv(x, t_emb, cond)
		self.connection = x
		if self.pools:
			x = self.pool(x)
		return x

	def up(self, x, t_emb, cond):
		if self.pools:
			x = self.upsampler(x)
		x = torch.cat((x,self.connection), dim=1) # concat channels

		for uconv in self.uconvs:
			x = uconv(x, t_emb, cond)

		return x