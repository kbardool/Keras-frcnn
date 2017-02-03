import numpy as np
import cv2
import random
import math
import data_augment


def get_img_output_length(width, height):
	def get_output_length(input_length):
		# zero_pad
		input_length += 6
		# apply 4 strided convolutions
		filter_sizes = [7, 3, 1, 1]
		stride = 2
		for filter_size in filter_sizes:
			input_length = (input_length - filter_size + stride) // stride
		return input_length

	return get_output_length(width), get_output_length(height)


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
		return x, y, w, h

	def intersection(a, b):
		x = max(a[0], b[0])
		y = max(a[1], b[1])
		w = min(a[2], b[2]) - x
		h = min(a[3], b[3]) - y
		if w < 0 or h < 0:
			return 0, 0, 0, 0
		return x, y, w, h

	i = intersection(a, b)
	u = union(a, b)

	area_i = i[2] * i[3]
	area_u = u[2] * u[3]
	return float(area_i) / float(area_u)


def get_new_img_size(width, height, img_min_side=600):
	if width <= height:
		f = float(img_min_side) / width
		resized_height = int(f * height)
		resized_width = img_min_side
	else:
		f = float(img_min_side) / height
		resized_width = int(f * width)
		resized_height = img_min_side

	return (resized_width, resized_height)


class SampleSelector:
	def __init__(self, class_count):
		# setting for data augmentation
		self.classes = class_count.keys()
		self.curr_class = 0
		self.num_classes = len(self.classes)

	def skip_sample_for_balanced_class(self, img_data):
		curr_class = self.classes[self.curr_class]

		class_in_img = False

		for bbox in img_data['bboxes']:

			cls_name = bbox['class']

			if cls_name == curr_class:
				class_in_img = True
				break

		if class_in_img:
			self.curr_class += 1

			if self.curr_class == self.num_classes:
				self.curr_class = 0

			return False
		else:
			return True


def get_anchor_gt(all_img_data, class_mapping, class_count, C):
	downscale = float(C.rpn_stride)

	anchor_sizes = C.anchor_box_scales
	anchor_ratios = C.anchor_box_ratios

	num_anchors = len(anchor_sizes) * len(anchor_ratios)

	sample_selector = SampleSelector(class_count)

	while True:
		# try:
		random.shuffle(all_img_data)

		for img_data in all_img_data:

			if C.balanced_classes and sample_selector.skip_sample_for_balanced_class(img_data):
				continue

			# read in image, and optionally add augmentation
			img_data, img = data_augment.augment(img_data, C)

			(width, height) = (img_data['width'], img_data['height'])
			(rows, cols, _) = img.shape

			assert cols == width
			assert rows == height

			# get image dimensions for resizing
			(resized_width, resized_height) = get_new_img_size(width, height)

			# resize the image so that smalles side is length = 600px
			img = cv2.resize(img, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)

			# calculate the output map size based on the network architecture
			(output_width, output_height) = get_img_output_length(resized_width, resized_height)

			# initialise empty output objectives
			Y_rpn_overlap = np.zeros((output_height, output_width, num_anchors))
			Y_is_box_valid = np.zeros((output_height, output_width, num_anchors))
			Y_rpn_regr = np.zeros((output_height, output_width, num_anchors * 4))

			# check overlap between anchor boxes and rois
			num_bboxes = len(img_data['bboxes'])
			num_anchors_for_bbox = [0] * num_bboxes
			best_anchor_for_bbox = [0] * num_bboxes
			best_iou_for_bbox = [0] * num_bboxes
			best_x_for_bbox = [[]] * num_bboxes
			best_dx_for_bbox = [[]] * num_bboxes

			pos_samples = []
			cls_samples = []
			neg_samples = []

			for ix in xrange(output_width):
				for jy in xrange(output_height):
					for anchor_size_idx, anchor_size in enumerate(anchor_sizes):
						for anchor_ratio_idx, anchor_ratio in enumerate(anchor_ratios):

							anchor_x = anchor_size * anchor_ratio[0]
							anchor_y = anchor_size * anchor_ratio[1]

							# coordinates of the current anchor box
							x1_anc = downscale * (ix + 0.5) - anchor_x / 2
							x2_anc = downscale * (ix + 0.5) + anchor_x / 2
							y1_anc = downscale * (jy + 0.5) - anchor_y / 2
							y2_anc = downscale * (jy + 0.5) + anchor_y / 2

							# ignore boxes that go across image boundaries
							if x1_anc < 0 or y1_anc < 0 or x2_anc > resized_width or y2_anc > resized_height:
								continue

							# bbox_type indicates whether an anchor should be a target 
							bbox_type = 'neg'

							for bbox_num, bbox in enumerate(img_data['bboxes']):
								# get the GT box coordinates, and resize to account for image resizing
								x1_gt = bbox['x1'] * (resized_width / float(width))
								x2_gt = bbox['x2'] * (resized_width / float(width))
								y1_gt = bbox['y1'] * (resized_height / float(height))
								y2_gt = bbox['y2'] * (resized_height / float(height))

								# calculate the regression targets
								tx = (x1_gt - x1_anc) / (x2_anc - x1_anc)
								ty = (y1_gt - y1_anc) / (y2_anc - y1_anc)
								tw = math.log((x2_gt - x1_gt) / (x2_anc - x1_anc))
								th = math.log((y2_gt - y1_gt) / (y2_anc - y1_anc))

								# get IOU of the current GT box and the current anchor box
								curr_iou = iou([x1_gt, y1_gt, x2_gt, y2_gt], [x1_anc, y1_anc, x2_anc, y2_anc])

								# all GT boxes should be mapped to an anchor box, so we keep track of which anchor box was best
								if curr_iou > best_iou_for_bbox[bbox_num]:
									# best_anchor_for_bbox[bbox_num] = [jy,ix,anchor_ratio_idx + 3 * anchor_size_idx]
									best_anchor_for_bbox[bbox_num] = [jy, ix, anchor_ratio_idx, anchor_size_idx]
									best_iou_for_bbox[bbox_num] = curr_iou
									best_x_for_bbox[bbox_num] = [x1_anc, x2_anc, y1_anc, y2_anc]
									best_dx_for_bbox[bbox_num] = [tx, ty, tw, th]

								# if the IOU is >0.3 and <0.7, it is ambiguous and no included in the objective
								if 0.3 < curr_iou < 0.7:
									# gray zone between neg and pos
									if bbox_type != 'pos':
										bbox_type = 'neutral'
								elif curr_iou > 0.7:
									# there may be multiple overlapping bboxes here
									bbox_type = 'pos'
									num_anchors_for_bbox[bbox_num] += 1
									best_regr = (tx, ty, tw, th)

								# samples for classification network
								if curr_iou < 0.1:
									# negative sample
									pass
								elif curr_iou < 0.5:
									# hard neg sample
									neg_samples.append((int(x1_anc / downscale), int(y1_anc / downscale),
														int((x2_anc - x1_anc) / downscale),
														int((y2_anc - y1_anc) / downscale)))
								else:
									# pos sample
									pos_samples.append((int(x1_anc / downscale), int(y1_anc / downscale),
														int((x2_anc - x1_anc) / downscale),
														int((y2_anc - y1_anc) / downscale)))
									cls_samples.append(bbox['class'])

							# turn on or off outputs depending on IOUs
							if bbox_type == 'neg':
								Y_is_box_valid[jy, ix, anchor_ratio_idx + 3 * anchor_size_idx] = 1
								Y_rpn_overlap[jy, ix, anchor_ratio_idx + 3 * anchor_size_idx] = 0
							elif bbox_type == 'neutral':
								Y_is_box_valid[jy, ix, anchor_ratio_idx + 3 * anchor_size_idx] = 0
								Y_rpn_overlap[jy, ix, anchor_ratio_idx + 3 * anchor_size_idx] = 0
							else:
								Y_is_box_valid[jy, ix, anchor_ratio_idx + 3 * anchor_size_idx] = 1
								Y_rpn_overlap[jy, ix, anchor_ratio_idx + 3 * anchor_size_idx] = 1
								Y_rpn_regr[jy, ix, 4 * anchor_ratio_idx + 4 * 3 * anchor_size_idx + 0] = best_regr[0]
								Y_rpn_regr[jy, ix, 4 * anchor_ratio_idx + 4 * 3 * anchor_size_idx + 1] = best_regr[1]
								Y_rpn_regr[jy, ix, 4 * anchor_ratio_idx + 4 * 3 * anchor_size_idx + 2] = best_regr[2]
								Y_rpn_regr[jy, ix, 4 * anchor_ratio_idx + 4 * 3 * anchor_size_idx + 3] = best_regr[3]

			# if a bbox doesnt have a corresponding anchor box, turn on the value with the greatest IOU
			for idx, num_anchor_for_bbox in enumerate(num_anchors_for_bbox):
				if num_anchor_for_bbox == 0:
					# no box with an IOU greater than zero ...
					if best_anchor_for_bbox[idx] == 0:
						continue
					Y_is_box_valid[
						best_anchor_for_bbox[idx][0], best_anchor_for_bbox[idx][1], best_anchor_for_bbox[idx][2] + 3 *
						best_anchor_for_bbox[idx][3]] = 1
					Y_rpn_overlap[
						best_anchor_for_bbox[idx][0], best_anchor_for_bbox[idx][1], best_anchor_for_bbox[idx][2] + 3 *
						best_anchor_for_bbox[idx][3]] = 1
					Y_rpn_regr[
						best_anchor_for_bbox[idx][0], best_anchor_for_bbox[idx][1], 4 * best_anchor_for_bbox[idx][
							2] + 4 * 3 * best_anchor_for_bbox[idx][3] + 0] = best_dx_for_bbox[idx][0]
					Y_rpn_regr[
						best_anchor_for_bbox[idx][0], best_anchor_for_bbox[idx][1], 4 * best_anchor_for_bbox[idx][
							2] + 4 * 3 * best_anchor_for_bbox[idx][3] + 1] = best_dx_for_bbox[idx][1]
					Y_rpn_regr[
						best_anchor_for_bbox[idx][0], best_anchor_for_bbox[idx][1], 4 * best_anchor_for_bbox[idx][
							2] + 4 * 3 * best_anchor_for_bbox[idx][3] + 2] = best_dx_for_bbox[idx][2]
					Y_rpn_regr[
						best_anchor_for_bbox[idx][0], best_anchor_for_bbox[idx][1], 4 * best_anchor_for_bbox[idx][
							2] + 4 * 3 * best_anchor_for_bbox[idx][3] + 3] = best_dx_for_bbox[idx][3]

			Y_rpn_overlap = np.transpose(Y_rpn_overlap, (2, 0, 1))
			Y_rpn_overlap = np.expand_dims(Y_rpn_overlap, axis=0)

			Y_is_box_valid = np.transpose(Y_is_box_valid, (2, 0, 1))
			Y_is_box_valid = np.expand_dims(Y_is_box_valid, axis=0)

			Y_rpn_regr = np.transpose(Y_rpn_regr, (2, 0, 1))
			Y_rpn_regr = np.expand_dims(Y_rpn_regr, axis=0)

			pos_locs = np.where(np.logical_and(Y_rpn_overlap[0, :, :, :] == 1, Y_is_box_valid[0, :, :, :] == 1))
			neg_locs = np.where(np.logical_and(Y_rpn_overlap[0, :, :, :] == 0, Y_is_box_valid[0, :, :, :] == 1))

			if len(pos_samples) == 0:
				continue

			pos_samples = np.array(pos_samples)
			neg_samples = np.array(neg_samples)
			cls_samples = np.array(cls_samples)

			target_pos_samples = C.num_rois / 2

			if pos_samples.shape[0] > target_pos_samples:
				val_locs = random.sample(range(pos_samples.shape[0]), target_pos_samples)
				valid_pos_samples = pos_samples[val_locs, :]
				valid_cls_samples = cls_samples[val_locs]
			else:
				valid_pos_samples = pos_samples
				valid_cls_samples = cls_samples

			val_locs = random.sample(range(neg_samples.shape[0]), C.num_rois - valid_cls_samples.shape[0])
			valid_neg_samples = neg_samples[val_locs, :]

			Y_rois = np.expand_dims(np.concatenate([valid_pos_samples, valid_neg_samples]), axis=0)
			Y_class_num = np.zeros((Y_rois.shape[1], len(class_mapping) + 1))

			for i in range(Y_rois.shape[1]):
				if i < valid_cls_samples.shape[0]:
					class_num = class_mapping[valid_cls_samples[i]]
					Y_class_num[i, class_num] = 1
				else:
					Y_class_num[i, -1] = 1

			Y_class_num = np.expand_dims(Y_class_num, axis=0)
			num_pos = len(pos_locs[0])

			if len(pos_locs[0]) > 128:
				val_locs = random.sample(range(len(pos_locs[0])), len(pos_locs[0]) - 128)
				Y_is_box_valid[0, pos_locs[0][val_locs], pos_locs[1][val_locs], pos_locs[2][val_locs]] = 0
				num_pos = 128

			if len(neg_locs[0]) + num_pos > 256:
				val_locs = random.sample(range(len(neg_locs[0])), len(neg_locs[0]) + num_pos - 256)
				Y_is_box_valid[0, neg_locs[0][val_locs], neg_locs[1][val_locs], neg_locs[2][val_locs]] = 0

			Y_rpn_cls = np.concatenate([Y_is_box_valid, Y_rpn_overlap], axis=1)
			Y_rpn_regr = np.concatenate([np.repeat(Y_rpn_overlap, 4, axis=1), Y_rpn_regr], axis=1)
			# if not np.sum(np.where(Y_rpn_regr[0,36:,:,:]!= 0)) == np.sum(np.where(Y_rpn_regr[0,:36,:,:]!= 0)):
			#	pdb.set_trace()
			img = np.transpose(img, (2, 0, 1))
			img = np.expand_dims(img, axis=0).astype('float32')
			img -= 127.5

			yield (img, Y_rpn_cls, Y_rpn_regr, Y_rois, Y_class_num)

		# except Exception as e:
		#	print(e)
		#	continue
