# Introduction
It is code for post of https://kimbring2.github.io/2020/01/06/al5d.html. Purpose of project is playing a stacking tower game using a Deep Learning methoh.

# Requirements
1. AL5D with BotBoarduino
2. USB webcam
3. Python package : pyserial, matplotlib, keras, opencv etc..
4. Pretrained model for depth estimation : You can a 'nyu.h5' file download link at https://github.com/ialhashim/DenseDepth.
5. It is good having a GPU in your computer. Because, I check monocular depth estimation process can require little GPU power for fast process.  

# Usage
1. Upload a AL5d_code.ino to your BotBoarduino
2. Run a AL5D_code.py in your terminal

# Trouble Shooting
1. USB port permission : sudo chmod a+rw /dev/ttyUSB0

# Work Plan
1. Making a basic RL environment where state, action, reward, done is existed for training
2. Implementing Model-Based RL algorithm for rolling a dice without virtual simulation 
