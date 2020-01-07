import os
import glob
import argparse
import matplotlib
import cv2 
import numpy as np

# Keras / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from keras.models import load_model
from layers import BilinearUpSampling2D
from utils import predict, load_images, display_images
from matplotlib import pyplot as plt

# Argument Parser
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--model', default='nyu.h5', type=str, help='Trained Keras model file.')
parser.add_argument('--input', default='examples/*.png', type=str, help='Input filename or folder.')
args = parser.parse_args()

# Custom object needed for inference and training
custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

print('Loading model...')

# Load model into GPU / CPU
model = load_model(args.model, custom_objects=custom_objects, compile=False)

print('\nModel loaded ({0}).'.format(args.model))

key = cv2.waitKey(1)
webcam = cv2.VideoCapture(0)
while True:
	check, frame = webcam.read()
	#cv2.imwrite(filename='saved_img_raw.png', img=frame)
	#print(check) #prints true as long as the webcam is running
	#print(frame) #prints matrix values of each framecd
	#print("frame.shape: " + str(frame.shape)) 
	#cv2.imshow("Capturing", frame)
	key = cv2.waitKey(1)

	# Input images
	#inputs = load_images(glob.glob(args.input))
	frame = frame / 255.0
	frame = np.clip(frame, 0, 1)
	frame = [frame]
	inputs = np.array(frame)
	#cv2.imshow("inputs", inputs)
	#print("inputs: " + str(inputs)) 
	
	#print('\nLoaded ({0}) images of size {1}.'.format(inputs.shape[0], inputs.shape[1:]))

	# Compute results
	outputs = predict(model, inputs)

	#matplotlib problem on ubuntu terminal fix
	#matplotlib.use('TkAgg')   

	#cv2.imshow("inputs[0]", inputs[0])
	# Display results
	viz = display_images(outputs.copy(), inputs.copy())
	viz = viz.astype(np.float32)
	viz = cv2.cvtColor(viz, cv2.COLOR_BGR2RGB)
	#print("viz.shape: " + str(viz.shape))
	cv2.imshow("Result", viz)
	#plt.figure(figsize=(10,5))
	#plt.imshow(outputs[0])
	#plt.imshow(viz)
	#plt.savefig('test.png')
	#plt.show()
	
