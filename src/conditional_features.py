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
Some torch modules to extract features that will be injected into our diffusion models.
'''

import torch
import math

def swish(x):
	return x*torch.sigmoid(x)

class SwishModule(torch.nn.Module):
	def __init__(self):
		super().__init__()
	def forward(self, x):
		return swish(x)

# simple MLP to transform flat states into a desired size
class StateVectorFeatures(torch.nn.Module):
	def __init__(self, initial_size=96, feature_size=256):
		super().__init__()
		self.fc1 = torch.nn.Linear(initial_size, feature_size)
		self.fc2 = torch.nn.Linear(feature_size, feature_size)

	def forward(self, x):
		return self.fc2(swish(self.fc1(x)))


# CNN Block
class CNNResPoolBlock(torch.nn.Module):
	def __init__(self, channels):
		super().__init__()
		self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size=5, stride=1, padding="same")
		self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size=5, stride=1, padding="same")
		self.drop = torch.nn.Dropout(0.1)
		self.pool = torch.nn.MaxPool2d(2)

	def forward(self, x):
		y = swish(self.conv1(x))
		y = swish(self.conv2(y))
		y = self.drop(y)
		return self.pool(x + y)


# Stack of multiple CNN blocks to turn an image into a flat feature vector
class CNNFeatures(torch.nn.Module):

	def __init__(self, initial_side=96, initial_channels=12, layers=4, channels=32, feature_size=256):
		super().__init__()

		self.initial_conv = torch.nn.Conv2d(initial_channels, channels, kernel_size=3, stride=1, padding="same")

		self.layers = torch.nn.ModuleList()
		side = initial_side
		for i in range(layers):
			self.layers.append(CNNResPoolBlock(channels))
			side = int(side/2)

		print("Flattened CNN size is:", side*side*channels)
		self.out1 = torch.nn.Linear(side*side*channels, feature_size*2)
		self.out2 = torch.nn.Linear(feature_size*2, feature_size)

	def forward(self, x):
		assert not torch.isnan(x).any(), "input contained nan"
		
		added_batch = False
		if len(x.shape)==3:
			added_batch = True
			x = x[None, :, :, :]

		x = swish(self.initial_conv(x))
		for layer in self.layers:
			x = layer(x)
		x = torch.flatten(x, start_dim=1)
		x = swish(self.out1(x))
		x = swish(self.out2(x))

		if added_batch:
			x = x[0]

		return x


# Sinusoidal encoding, for embedding time
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


# Embed time and pass through one hidden layer
class TimeFeatures(torch.nn.Module):
	def __init__(self, times=1000, feature_size=256):
		super().__init__()
		self.pos_emb = PositionalEncoding(feature_size, times)
		self.fc1 = torch.nn.Linear(feature_size, feature_size)

	def forward(self, x):
		x = self.pos_emb(x)
		x = swish(self.fc1(x))
		return x
