import os
import random
import xml.etree.ElementTree as ET
import math
import pprint
import pdb
import cv2
import json
import numpy as np
import sys
from keras_frcnn import config

sys.setrecursionlimit(40000)
C = config.Config()
C.use_horizontal_flips = False
C.use_vertical_flips = False


def format_img(img):
	img_min_side = 600.0
	(height,width,_) = img.shape
	
	if width <= height:
		f = img_min_side/width
		new_height = int(f * height)
		new_width = int(img_min_side)
	else:
		f = img_min_side/height
		new_width = int(f * width)
		new_height = int(img_min_side)
	img = cv2.resize(img,(new_width,new_height), interpolation=cv2.INTER_CUBIC)
	img = img[:,:,(2,1,0)]
	img = np.transpose(img,(2,0,1)).astype(np.float32)
	img = np.expand_dims(img, axis=0)
	img[:, 0, :, :] -= 103.939
	img[:, 1, :, :] -= 116.779
	img[:, 2, :, :] -= 123.68
	return img

with open('classes.json', 'r') as class_data_json:
    class_mapping = json.load(class_data_json)

if 'bg' not in class_mapping:
	class_mapping['bg'] = len(class_mapping)

class_mapping = {v: k for k, v in class_mapping.iteritems()}

class_to_color = {class_mapping[v]:np.random.randint(0,255,3) for v in class_mapping}
num_rois = 16

import keras_frcnn.resnet as nn
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras_frcnn import roi_helpers

if K.image_dim_ordering() == 'th':
	input_shape_img = (3, None, None)
	input_shape_features = (1024, None, None)
else:
	input_shape_img = (None, None, 3)
	input_shape_features = (None, None, 1024)


img_input = Input(shape=input_shape_img)

feature_map_input = Input(shape=input_shape_features)

roi_input = Input(shape=(num_rois, 4))

# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input)

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn = nn.rpn(shared_layers,num_anchors)

# classifier, uses base layers + proposals
print(class_mapping)

classifier = nn.classifier(feature_map_input,roi_input,num_rois,nb_classes=len(class_mapping))

model_rpn = Model(img_input,rpn + [shared_layers])
model_classifier = Model([feature_map_input,roi_input],classifier)

weights_path = 'model_frcnn.hdf5'

model_rpn.load_weights(weights_path, by_name=True)
model_classifier.load_weights(weights_path, by_name=True)

model_rpn.compile(optimizer='sgd',loss='mse')
model_classifier.compile(optimizer='sgd',loss='mse')

all_imgs = []

classes = {}

visualise = True

print('Parsing annotation files')
img_path = sys.argv[1]
bufsize = 0

for idx,img_name in enumerate(sorted(os.listdir(img_path))):
	print(img_name)
	filepath = os.path.join(img_path,img_name)
	img = cv2.imread(filepath)

	X = format_img(img)

	img_scaled = np.transpose(X[0,(2,1,0),:,:],(1,2,0)).copy()
	img_scaled[:, :, 0] += 123.68
	img_scaled[:, :, 1] += 116.779
	img_scaled[:, :, 2] += 103.939
	
	img_scaled = img_scaled.astype(np.uint8)

	if K.image_dim_ordering() == 'tf':
		X = np.transpose(X,(0,2,3,1))
	# get the feature maps and output from the RPN
	[Y1, Y2, F] = model_rpn.predict(X)

	R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.image_dim_ordering())

	# convert from (x1,y1,x2,y2) to (x,y,w,h)
	R[:,2] = R[:,2] - R[:,0]
	R[:,3] = R[:,3] - R[:,1]

	# apply the spatial pyramid pooling to the proposed regions
	bboxes = {}
	probs = {}
	for jk in range(R.shape[0]//num_rois + 1):
		ROIs = np.expand_dims(R[num_rois*jk:num_rois*(jk+1),:],axis=0)
		if ROIs.shape[1] == 0:
			break

		if jk == R.shape[0]//num_rois:
			#pad R
			curr_shape = ROIs.shape
			target_shape = (curr_shape[0],num_rois,curr_shape[2])
			ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
			ROIs_padded[:,:curr_shape[1],:] = ROIs
			ROIs_padded[0,curr_shape[1]:,:] = ROIs[0,0,:]
			ROIs = ROIs_padded

		[P_cls, P_regr] = model_classifier.predict([F, ROIs])
		P_regr = P_regr / C.std_scaling

		for ii in range(P_cls.shape[1]):

			if np.max(P_cls[0,ii,:]) < 0.5 or np.argmax(P_cls[0,ii,:]) == (P_cls.shape[2] - 1):
				continue

			cls_name = class_mapping[np.argmax(P_cls[0,ii,:])]

			if cls_name not in bboxes:
				bboxes[cls_name] = []
				probs[cls_name] = []

			(x, y, w, h) = ROIs[0,ii,:]

			cls_num = np.argmax(P_cls[0, ii, :])
			(tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
			x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)

			bboxes[cls_name].append([16*x, 16*y, 16*(x+w), 16*(y+h)])
			probs[cls_name].append(np.max(P_cls[0, ii, :]))

	all_dets = {}

	for key in bboxes:
		bbox = np.array(bboxes[key])

		new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlapThresh=0.5)
		for jk in range(new_boxes.shape[0]):
			(x1,y1,x2,y2) = new_boxes[jk,:]

			cv2.rectangle(img_scaled,(x1, y1), (x2, y2), class_to_color[key],2)

			textLabel = '{}:{}'.format(key,int(100*new_probs[jk]))
			if key not in all_dets:
				all_dets[key] = 100*new_probs[jk]
			else:
				all_dets[key] = max(all_dets[key],100*new_probs[jk])

			(retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,1,1)
			textOrg = (x1,y1+20)

			#cv2.rectangle(img_scaled,(textOrg[0] - 5,textOrg[1]+baseLine - 5),(textOrg[0]+retval[0] + 5,textOrg[1]-retval[1] - 5),(0,0,0),2)
			#cv2.rectangle(img_scaled,(textOrg[0] - 5,textOrg[1]+baseLine - 5),(textOrg[0]+retval[0] + 5,textOrg[1]-retval[1] - 5),(255,255,255),-1)
			#cv2.putText(img_scaled,textLabel,textOrg,cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),1)
	cv2.imshow('img',img_scaled)
	cv2.waitKey(0)
