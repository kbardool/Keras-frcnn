
class Config:

	def __init__(self):
		# setting for data augmentation
		self.use_horizontal_flips = True
		self.use_vertical_flips = True
		self.scale_augment = False
		self.random_rotate = False
		self.random_rotate_scale = 15.

		# anchor box scales
		self.anchor_box_scales = [128, 256, 512] #[128, 256, 512] 
		# anchor box ratios
		self.anchor_box_ratios = [[1,1], [1,2], [2,1]] #[[1, 1], [1, 2], [2, 1]]
		

		# size to resize the smallest size of the image
		self.im_size = 400#600

		# number of ROIs at once
		self.num_rois = 2

		# stride at the RPN (this depends on the network configuration)
		self.rpn_stride = 16

		self.balanced_classes = True
