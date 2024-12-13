# Diffusion Policies Implementation

This is an implementation of Diffusion Policies (DDPM and DDIM), Flow Matching Policies, and Shortcut Models, tested on the CarRacing-v3 environment. It is trained from 20 trajectories of a human expert playing the game with a joystick.

This work was authored by Ted Staley and is Copyright © 2024 The Johns Hopkins University Applied Physics Laboratory LLC, please see the LICENSE file.

Full accompanying notes on this repo and the technique can be found here:
TODO


### Brief Outline of Files

The key files of interest will be:

- /data_collect/ : contains expert data and a script to collect more
- /src/backbones/unet.py : the network architecture (1D Conditional UNet, optionally with shortcut)
- /src/algos.py : unique aspects of each algorithm
- /src/train.py : trains the diffusion policy or flow matching model
- /src/train_shortcut.py : trains a shortcut flow matching model
- /src/deploy.py : runs a saved model



### Results

The train.py script takes about 8 hours to run on a 4090 GPU and converges to near-expert level with occasional poor performance. The expert data has an average score of 923 while the trained model scores 889 on average, and 920 if the bottom 10% of runs are discarded (i.e. most runs match expert level). The training curve over 500k batches converges to an MSE of about 0.0015 for diffusion:

![learning-curve](./assets/learning.png)


Performance of the trained policies are nearly identical to the above for flow matching and shortcut models.

Notably, the shortcut model can match this performance even with one-step inference, which allows it to run at 60 FPS or more.


### Video of Results:

![results](./assets/three_animation.gif)


