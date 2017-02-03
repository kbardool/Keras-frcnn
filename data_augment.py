import cv2
import pdb
import numpy as np


def augment(img_data, config):
	assert 'filepath' in img_data
	assert 'bboxes' in img_data
	assert 'width' in img_data
	assert 'height' in img_data

	img = cv2.imread(img_data['filepath'])
	# print(img_data['filepath'])
	rows, cols = img.shape[:2]

	if config.use_horizontal_flips:
		if np.random.randint(0, 2) == 0:
			img = cv2.flip(img, 1)
			for bbox in img_data['bboxes']:
				x1 = bbox['x1']
				x2 = bbox['x2']
				bbox['x2'] = cols - x1
				bbox['x1'] = cols - x2
	if config.use_vertical_flips:
		if np.random.randint(0, 2) == 0:
			img = cv2.flip(img, 0)
			for bbox in img_data['bboxes']:
				y1 = bbox['y1']
				y2 = bbox['y2']
				bbox['y2'] = rows - y1
				bbox['y1'] = rows - y2

	for bbox in img_data['bboxes']:
		x1 = bbox['x1']
		x2 = bbox['x2']
		y1 = bbox['y1']
		y2 = bbox['y2']
	# cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255))
	# cv2.imshow('a',img)
	# cv2.waitKey(0)
	return img_data, img
