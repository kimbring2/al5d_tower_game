import os
import glob
import argparse
import matplotlib
import cv2 
import numpy as np
import serial
from matplotlib import pyplot as plt

# Mask R-CNN
import mrcnn.utils as mrcnn_utils
import mrcnn.visualize as mrcnn_visualize
import mrcnn.model as mrcnn_modellib

# Tower Game Mask R-CNN main
from samples.tower import tower

# Keras / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
import tensorflow as tf
from keras.models import load_model
from layers import BilinearUpSampling2D

# Monocular Depth Estimation 
import depth_utils

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

# Argument Parser
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--model', default='/home/kimbring2/nyu.h5', type=str, help='Trained Keras model file.')
parser.add_argument('--input', default='examples/*.png', type=str, help='Input filename or folder.')
args = parser.parse_args()

# Custom object needed for inference and training
custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

print('Loading model...')

# Load Depth Estimation model into GPU / CPU
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

tower_config = tower.TowerConfig()
TOWER_DIR = "/home/kimbring2/DenseDepth/main/tower"

# Load validation dataset
dataset = tower.TowerDataset()
dataset.load_tower(TOWER_DIR, "val")

# Must call before using the dataset
dataset.prepare()

print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

# Load MRCNN weights
print("Loading weights ", RCNN_MODEL_DIR)
model_rcnn.load_weights(RCNN_MODEL_DIR, by_name=True)

key = cv2.waitKey(1)
webcam = cv2.VideoCapture(0)
while True:
	check, frame = webcam.read()
	frame_rcnn = frame
	key = cv2.waitKey(1)

	# Input images
	frame = frame / 255.0
	frame = np.clip(frame, 0, 1)
	# frame.shape: (480, 640, 3)

	frame = [frame]
	inputs = np.array(frame)
	
	# Run object detection
	inputs_rcnn = [frame_rcnn]
	mrcnn_results = model_rcnn.detect(inputs_rcnn, verbose=0)
	#print("mrcnn_results: " + str(mrcnn_results))
	
	# Display MRCNN results
	ax = get_ax(1)
	r = mrcnn_results[0]
	#print("r['rois']: " + str(r['rois']))
	#print("r['masks']: " + str(r['masks']))
	#print("r['class_ids']: " + str(r['class_ids']))
	#print("r['scores']: " + str(r['scores']))

	mrcnn_outputs = mrcnn_visualize.display_instances(inputs_rcnn[0], r['rois'], r['masks'], r['class_ids'], 
	                				        		  dataset.class_names, r['scores'], ax=ax,
		                    						  title="Predictions")
	#mrcnn_outputs_resize = cv2.resize(mrcnn_outputs, (int(240), int(320)))
	cv2.imshow("Mask RCNN Output", mrcnn_outputs)

	# Compute Depth Estimation results
	depth_outputs = depth_utils.predict(model_depth, inputs)
	depth_outputs_np = np.array(depth_outputs)
	#print("depth_outputs.shape: " + str(depth_outputs.shape))
 
	# Display Depth Estimation results
	depth_viz = depth_utils.display_images(depth_outputs.copy(), inputs.copy())
	depth_viz = depth_viz.astype(np.float32)
	depth_viz = cv2.cvtColor(depth_viz, cv2.COLOR_BGR2RGB)
	depth_viz = depth_viz[0:0+240, 320:320+320]
	#print("depth_viz: " + str(depth_viz))
	# depth_viz.shape: (240, 640, 3)

	for idx, roi in enumerate(r['rois']):
		y1, x1, y2, x2 = roi
		y1 = int(y1 / 2.0) 
		x1 = int(x1 / 2.0) 
		y2 = int(y2 / 2.0) 
		x2 = int(x2 / 2.0)
		x_middle = int((x1 + x2) / 2)
		y_middle = int((y1 + y2) / 2)

		print("x_middle: " + str(x_middle))
		print("y_middle: " + str(y_middle))
		print("depth: " + str(depth_outputs[0][y_middle,x_middle]))
		print("object name: " + str(dataset.class_names[r['class_ids'][idx]]))
		print("")
		depth_viz = cv2.rectangle(depth_viz, (x1, y1), (x2, y2), (0,255,0), 1)

	cv2.imshow("Depth Estimation Result", depth_viz)