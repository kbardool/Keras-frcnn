import cv2
import numpy as np
import copy

def augment(img_data, config, augment=True):
	assert 'filepath' in img_data
	assert 'bboxes' in img_data
	assert 'width' in img_data
	assert 'height' in img_data

	img_data_aug = copy.deepcopy(img_data)

	img = cv2.imread(img_data_aug['filepath'])

	if augment:
		rows, cols = img.shape[:2]

		if config.use_horizontal_flips and np.random.randint(0, 2) == 0:
			img = cv2.flip(img, 1)
			for bbox in img_data_aug['bboxes']:
				x1 = bbox['x1']
				x2 = bbox['x2']
				bbox['x2'] = cols - x1
				bbox['x1'] = cols - x2

		if config.use_vertical_flips and np.random.randint(0, 2) == 0:
			img = cv2.flip(img, 0)
			for bbox in img_data_aug['bboxes']:
				y1 = bbox['y1']
				y2 = bbox['y2']
				bbox['y2'] = rows - y1
				bbox['y1'] = rows - y2


		if config.random_rotate:
			M = cv2.getRotationMatrix2D((cols/2, rows/2), np.random.randint(-config.random_rotate_scale, config.random_rotate_scale), 1)
			img = cv2.warpAffine(img, M, (cols, rows), flags=cv2.INTER_CUBIC, borderMode= cv2.BORDER_REPLICATE)
			for bbox in img_data_aug['bboxes']:
				K = np.array([[bbox['x1'],bbox['y1']],[bbox['x2'],bbox['y2']],[bbox['x1'],bbox['y2']],[bbox['x2'],bbox['y1']]])
				K = cv2.transform(K.reshape(4,1,2),M)[:,0,:]

				(x1, y1) = np.min(K, axis=0)
				(x2, y2) = np.max(K, axis=0)

				bbox['x1'] = x1
				bbox['x2'] = x2
				bbox['y1'] = y1
				bbox['y2'] = y2

	return img_data_aug, img
