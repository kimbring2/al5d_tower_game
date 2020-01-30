# Introduction
It is code for post of https://kimbring2.github.io/2020/01/06/al5d.html. Purpose of project is playing a stacking tower game using a Deep Learning methoh.

# Requirements
1. Lynxmotion AL5D robot arm with BotBoarduino and PS2 Controller V4
2. USB webcam
3. Python package : pyserial, matplotlib, keras, opencv etc..
4. Pretrained model for depth estimation : You can a 'nyu.h5' file download link at https://github.com/ialhashim/DenseDepth.
5. It is good having a GPU in your computer. Because, I check monocular depth estimation process can require little GPU power for fast process.  
6. Mask R-CNN model is used for detection and segmentaion of game object. You need to install install a requirement at https://github.com/matterport/Mask_RCNN for using model.
7. Mask R-CNN pretrined model for tower game object : https://drive.google.com/open?id=1IPUHQdlMZwtIHkCuH5I3SMtnYkCClWvb

# Usage
1. Upload a 'AL5d_code.ino' to your BotBoarduino : you can control a each motor of robot arm using a Arduino terminal command or PS2 controller
2. Type a 'python AL5D_code.py' for controlling robot and watching Webcam image in your terminal
3. Open a 'inspect_tower_data.ipynb' and run it for checking dataset
4. Type a 'python3 tower.py train --dataset=/path/to/tower/dataset --weights=coco' for traning model
5. Open a 'inspect_tower_data.ipynb' and run it for checking trained model

# Trouble Shooting
1. USB port permission : sudo chmod a+rw /dev/ttyUSB0

# Work Plan
1. Making a basic RL environment where state, action, reward, done is existed for training
2. Implementing Model-Based RL algorithm for grasping a block without virtual simulation 
