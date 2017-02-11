from keras import backend as K

class Config:

	def __init__(self):
		# setting for data augmentation
		self.use_horizontal_flips = True
		self.use_vertical_flips = False
		self.scale_augment = False
		self.random_rotate = False
		self.random_rotate_scale = 15.

		# anchor box scales
		self.anchor_box_scales = [128, 256, 512]

		# anchor box ratios
		self.anchor_box_ratios = [[1, 1], [1, 2], [2, 1]]

		# size to resize the smallest side of the image
		self.im_size = 600

		# number of ROIs at once
		self.num_rois = 2

		# stride at the RPN (this depends on the network configuration)
		self.rpn_stride = 16

		self.balanced_classes = True
		
		#location of pretrained weights for the base network 
		if K.image_dim_ordering() == 'th':
			self.base_net_weights = './resnet50_weights_th_dim_ordering_th_kernels.h5'
		else:
			self.base_net_weights = './resnet50_weights_tf_dim_ordering_tf_kernels.h5'
		
		
		self.model_path = 'model_frcnn.hdf5'
