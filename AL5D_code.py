import os
import glob
import argparse
import matplotlib
import cv2 
import numpy as np
import serial

import tensorflow as tf


#from mrcnn import utils
import mrcnn.utils as mrcnn_utils

#from mrcnn import visualize
#from mrcnn.visualize import display_images
import mrcnn.visualize as mrcnn_visualize
import mrcnn.model as mrcnn_modellib
from mrcnn.model import log as mrcnn_log
#import mrcnn.model.log as mrcnn_log

from samples.tower import tower

# For communication with Botboduino of AL5D
ser = serial.Serial('/dev/ttyUSB0')
#ser.write(b'w') # Base up action
#ser.write(b's') # Base down action
#ser.write(b'a') # Shldr up action
#ser.write(b'd') # Shldr down action
#ser.write(b'e') # Elb up action
#ser.write(b'q') # Elb down action
#ser.write(b'r') # Wrist up action
#ser.write(b't') # Wrist down action
#ser.write(b'z') # Gripper open action
#ser.write(b'x') # Gripper close action

# Keras / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from keras.models import load_model
from layers import BilinearUpSampling2D
import depth_utils
#from utils import predict, load_images, display_images
from matplotlib import pyplot as plt

# Argument Parser
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--model', default='/home/kimbring2/nyu.h5', type=str, help='Trained Keras model file.')
parser.add_argument('--input', default='examples/*.png', type=str, help='Input filename or folder.')
args = parser.parse_args()

# Custom object needed for inference and training
custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

print('Loading model...')

# Load model into GPU / CPU
model_depth = load_model(args.model, custom_objects=custom_objects, compile=False)

print('\nModel loaded ({0}).'.format(args.model))

config_rcnn = tower.TowerConfig()
class InferenceConfig(config_rcnn.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

depth_config = InferenceConfig()

DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0
TEST_MODE = "inference"
RCNN_MODEL_DIR = "/home/kimbring2/DenseDepth/main/mask_rcnn_tower.h5"

with tf.device(DEVICE):
    model_rcnn = mrcnn_modellib.MaskRCNN(mode="inference", model_dir=RCNN_MODEL_DIR,
                              config=depth_config)

#weights_path = model.find_last()

tower_config = tower.TowerConfig()
TOWER_DIR = "/home/kimbring2/DenseDepth/main/tower"

# Load validation dataset
dataset = tower.TowerDataset()
dataset.load_tower(TOWER_DIR, "val")

# Must call before using the dataset
dataset.prepare()

print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

# Load weights
print("Loading weights ", RCNN_MODEL_DIR)
model_rcnn.load_weights(RCNN_MODEL_DIR, by_name=True)

key = cv2.waitKey(1)
webcam = cv2.VideoCapture(0)
while True:
	check, frame = webcam.read()
	#cv2.imwrite(filename='saved_img_raw.png', img=frame)
	#print(check) #prints true as long as the webcam is running
	#print(frame) #prints matrix values of each framecd
	#print("frame.shape: " + str(frame.shape)) 
	#cv2.imshow("Capturing", frame)
	frame_rcnn = frame
	key = cv2.waitKey(1)

	# Input images
	#inputs = load_images(glob.glob(args.input))
	frame = frame / 255.0
	frame = np.clip(frame, 0, 1)
	frame = [frame]
	inputs = np.array(frame)
	#cv2.imshow("inputs", inputs)
	#print("inputs.shape: " + str(inputs.shape)) 
	
	#print('\nLoaded ({0}) images of size {1}.'.format(inputs.shape[0], inputs.shape[1:]))

	# Run object detection
	inputs_rcnn = [frame_rcnn]
	results = model_rcnn.detect(inputs_rcnn, verbose=1)
	#print("results: " + str(results))
	
	# Display results
	ax = get_ax(1)
	r = results[0]
	mrcnn_visualize.display_instances(inputs_rcnn[0], r['rois'], r['masks'], r['class_ids'], 
		                    	dataset.class_names, r['scores'], ax=ax,
		                    	title="Predictions")
	# Compute results
	outputs = depth_utils.predict(model_depth, inputs)

	#matplotlib problem on ubuntu terminal fix
	#matplotlib.use('TkAgg')   

	#cv2.imshow("inputs[0]", inputs[0])
	# Display results
	viz = depth_utils.display_images(outputs.copy(), inputs.copy())
	viz = viz.astype(np.float32)
	viz = cv2.cvtColor(viz, cv2.COLOR_BGR2RGB)
	#print("viz.shape: " + str(viz.shape))
	cv2.imshow("Result", viz)
	#plt.figure(figsize=(10,5))
	#plt.imshow(outputs[0])
	#plt.imshow(viz)
	#plt.savefig('test.png')
	#plt.show()