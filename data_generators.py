import numpy as np
import cv2
import random
import pdb

def get_img_output_length(width,height):
	def get_output_length(input_length):
		# zero_pad
		input_length += 6
		# apply 4 srided convolutions
		filter_sizes = [7,3,1,1]
		stride = 2
		for filter_size in filter_sizes:
			input_length = (input_length - filter_size + stride) // stride
		return input_length
	return(get_output_length(width),get_output_length(height))


def iou(a, b):
	# a and b should be (x1,y1,x2,y2)
	assert a[0] < a[2]
	assert a[1] < a[3]
	assert b[0] < b[2]
	assert b[1] < b[3]

	def union(a, b):
		x = min(a[0], b[0])
		y = min(a[1], b[1])
		w = max(a[2], b[2]) - x
		h = max(a[3], b[3]) - y
		return (x, y, w, h)

	def intersection(a, b):
		x = max(a[0], b[0])
		y = max(a[1], b[1])
		w = min(a[2], b[2]) - x
		h = min(a[3], b[3]) - y
		if w < 0 or h < 0: 
			return (0, 0, 0, 0)
		return (x, y, w, h)

	i = intersection(a, b)
	u = union(a, b)

	area_i = i[2] * i[3]
	area_u = u[2] * u[3]
	return float(area_i)/float(area_u)


def get_new_img_size(width,height,img_min_side = 600):
	
	if width <= height:
		f = float(img_min_side)/width
		new_height = int(f * height)
		new_width = img_min_side
	else:
		f = float(img_min_side)/height
		new_width = int(f * width)
		new_height = img_min_side
		
	return(new_width,new_height)

def get_anchor_gt(all_imgs,class_mapping,num_rois=2):

	downscale = 16.0

	anchor_sizes = [128, 256, 512]
	anchor_ratios = [[1,1],[1,2],[2,1]]

	#anchor_sizes_downscaled = [int(s/downscale) for s in anchor_sizes]
	num_anchors = len(anchor_sizes) * len(anchor_ratios)

	vis = False

	while True:
		try:
			random.shuffle(all_imgs)

			for img in all_imgs:
				cv2img = cv2.imread(img['filepath'])
				if vis:
					for bbox in img['bboxes']:
						x1g = bbox['x1']
						x2g = bbox['x2']
						y1g = bbox['y1']
						y2g = bbox['y2']
						cv2.rectangle(cv2img,(x1g,y1g),(x2g,y2g),(0,0,255))

				(width,height) = (img['width'],img['height'])
				(rows,cols,_) = cv2img.shape

				assert cols == width
				assert rows == height
				
				(new_width,new_height) = get_new_img_size(width,height)
				
				cv2img = cv2.resize(cv2img,(new_width,new_height),interpolation = cv2.INTER_CUBIC)

				(width_downscaled,height_downscaled) = get_img_output_length(new_width,new_height)

				output_overlap = np.zeros((height_downscaled,width_downscaled,num_anchors))
				output_valid_box = np.zeros((height_downscaled,width_downscaled,num_anchors))

				# check overlap between anchor boxes and rois
				num_bboxes = len(img['bboxes'])
				num_anchors_for_bbox = [0] * num_bboxes
				best_anchor_for_bbox = [0] * num_bboxes
				best_iou_for_bbox = [0] * num_bboxes
				best_x_for_bbox = [[]] * num_bboxes

				pos_samples = []
				cls_samples = []
				neg_samples = []

				for ix in xrange(width_downscaled):
					for jy in xrange(height_downscaled):
						for anchor_size_idx,anchor_size in enumerate(anchor_sizes):
							for anchor_ratio_idx,anchor_ratio in enumerate(anchor_ratios):
								anchor_x = anchor_size * anchor_ratio[0]
								anchor_y = anchor_size * anchor_ratio[1]
								x1_anc = downscale * (ix + 0.5) - anchor_x/2
								x2_anc = downscale * (ix + 0.5) + anchor_x/2
								y1_anc = downscale * (jy + 0.5) - anchor_y/2
								y2_anc = downscale * (jy + 0.5) + anchor_y/2
								max_iou = 0.0

								if x1_anc < 0 or y1_anc < 0 or x2_anc > new_width or y2_anc > new_height:
									continue

								bbox_type = 'neg'

								for bbox_num,bbox in enumerate(img['bboxes']):
									x1_gt = bbox['x1'] * (new_width/float(width))
									x2_gt = bbox['x2'] * (new_width/float(width))
									y1_gt = bbox['y1'] * (new_height/float(height))
									y2_gt = bbox['y2'] * (new_height/float(height))

									curr_iou = iou([x1_gt,y1_gt,x2_gt,y2_gt],[x1_anc,y1_anc,x2_anc,y2_anc])

									if curr_iou > best_iou_for_bbox[bbox_num]:
										best_anchor_for_bbox[bbox_num] = [jy,ix,anchor_ratio_idx + 3 * anchor_size_idx]
										best_iou_for_bbox[bbox_num] = curr_iou
										best_x_for_bbox[bbox_num] = [x1_anc,x2_anc,y1_anc,y2_anc]

									if curr_iou > 0.3 and curr_iou < 0.7:
										# gray zone between neg and pos
										if bbox_type != 'pos':
											bbox_type = 'neutral'
									elif curr_iou > 0.7:
										# there may be multiple overlapping bboxes here
										bbox_type = 'pos'
										num_anchors_for_bbox[bbox_num] += 1

									# samples for classification network
									if curr_iou < 0.1:
										# negative sample
										pass
									elif curr_iou < 0.5:
										# hard neg sample
										neg_samples.append((int(x1_anc/downscale),int(y1_anc/downscale),int((x2_anc-x1_anc)/downscale),int((y2_anc - y1_anc)/downscale)))
									else:
										# pos sample
										pos_samples.append((int(x1_anc/downscale),int(y1_anc/downscale),int((x2_anc-x1_anc)/downscale),int((y2_anc - y1_anc)/downscale)))
										cls_samples.append(bbox['class'])
								
								if bbox_type == 'neg':
									output_valid_box[jy,ix,anchor_ratio_idx + 3 * anchor_size_idx] = 1
									output_overlap[jy,ix,anchor_ratio_idx + 3 * anchor_size_idx] = 0
								elif bbox_type == 'neutral':
									output_valid_box[jy,ix,anchor_ratio_idx + 3 * anchor_size_idx] = 0
								else:
									output_valid_box[jy,ix,anchor_ratio_idx + 3 * anchor_size_idx] = 1
									output_overlap[jy,ix,anchor_ratio_idx + 3 * anchor_size_idx] = 1

				for idx,num_anchor_for_bbox in enumerate(num_anchors_for_bbox):
					if num_anchor_for_bbox == 0:
						# no box with an IOU greater than zero ...
						if best_anchor_for_bbox[idx] == 0:
							continue
						output_valid_box[best_anchor_for_bbox[idx][0],best_anchor_for_bbox[idx][1],best_anchor_for_bbox[idx][2]] = 1
						output_overlap[best_anchor_for_bbox[idx][0],best_anchor_for_bbox[idx][1],best_anchor_for_bbox[idx][2]] = 1
						if vis:
							cv2.rectangle(cv2img,(int(best_x_for_bbox[idx][0]* downscale / fx),int(best_x_for_bbox[idx][2]* downscale / fx)),(int(best_x_for_bbox[idx][1]* downscale / fx),int(best_x_for_bbox[idx][3]* downscale / fx)),(0,255,0),2)

				output_overlap = np.transpose(output_overlap,(2,0,1))
				output_overlap = np.expand_dims(output_overlap,axis = 0)

				output_valid_box = np.transpose(output_valid_box,(2,0,1))
				output_valid_box = np.expand_dims(output_valid_box,axis = 0)

				pos_locs = np.where(np.logical_and(output_overlap[0,:,:,:] == 1,output_valid_box[0,:,:,:] == 1))
				neg_locs = np.where(np.logical_and(output_overlap[0,:,:,:] == 0,output_valid_box[0,:,:,:] == 1))		

				if len(pos_samples) == 0:
					continue

				pos_samples = np.array(pos_samples)
				neg_samples = np.array(neg_samples)
				cls_samples = np.array(cls_samples)

				target_pos_samples = num_rois - 2

				if pos_samples.shape[0] > target_pos_samples:
					val_locs = random.sample(range(pos_samples.shape[0]), target_pos_samples)
					valid_pos_samples = pos_samples[val_locs,:]
					valid_cls_samples = cls_samples[val_locs]
				else:
					valid_cls_samples = cls_samples
					valid_pos_samples = pos_samples

				val_locs = random.sample(range(neg_samples.shape[0]),num_rois - valid_cls_samples.shape[0])
				valid_neg_samples = neg_samples[val_locs,:]

				classifier_samples = np.expand_dims(np.concatenate([valid_pos_samples,valid_neg_samples]),axis=0)
				classifier_Y = np.zeros((classifier_samples.shape[1],len(class_mapping)+1))

				for i in range(classifier_samples.shape[1]):
					if i < valid_cls_samples.shape[0]:
						class_num = class_mapping[valid_cls_samples[i]]
						classifier_Y[i,class_num] = 1
					else:
						classifier_Y[i,-1] = 1
						
				classifier_Y = np.expand_dims(classifier_Y,axis=0)
				num_pos = len(pos_locs[0])
				
				if len(pos_locs[0]) > 128:
					val_locs = random.sample(range(len(pos_locs[0])), len(pos_locs[0]) - 128)
					output_valid_box[0,pos_locs[0][val_locs],pos_locs[1][val_locs],pos_locs[2][val_locs]] = 0
					num_pos = 128

				if len(neg_locs[0]) + num_pos > 256:
					val_locs = random.sample(range(len(neg_locs[0])),len(neg_locs[0]) + num_pos - 256)
					output_valid_box[0,neg_locs[0][val_locs],neg_locs[1][val_locs],neg_locs[2][val_locs]] = 0
				
				output_rpn_cls = np.concatenate([output_valid_box,output_overlap],axis = 1)
				output_rpn_regr = np.zeros((1,2 * 4 * num_anchors,height_downscaled,width_downscaled,))

				cv2img = np.transpose(cv2img,(2,0,1))
				cv2img = np.expand_dims(cv2img,axis = 0).astype('float32')
				cv2img -= 127.0

				if vis:
					cv2.imshow('img',cv2img)

					cv2.waitKey(0)
				yield(cv2img,output_rpn_cls,output_rpn_regr,classifier_samples,classifier_Y)

		except Exception as e:
			print(e)
			continue

