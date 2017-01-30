import cv2
import pdb

def augment(img_data,config):
	print(img_data)
	assert 'filepath' in img_data
	assert 'bboxes' in img_data
	assert 'width' in img_data
	assert 'height' in img_data

	img = cv2.imread(img_data['filepath'])
	rows,cols = img.shape[:2]

	if config.use_horizontal_flips:
		img = cv2.flip(img,1)
		for bbox in img_data['bboxes']:
			x1 = bbox['x1']
			x2 = bbox['x2']
			bbox['x2'] = cols - bbox['x1']
			bbox['x1'] = cols - bbox['x2']
	if config.use_vertical_flips:
		print('Warning: not yet implemented')
	print(img_data)
	pdb.set_trace()
	return img_data