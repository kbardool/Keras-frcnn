import random
import pprint
import sys
import json
import config
import pascal_voc_parser as parser

sys.setrecursionlimit(40000)

C = config.Config()
C.num_rois = 2

<<<<<<< HEAD
#import pascal_voc_parser as parser
#all_imgs,classes_count,class_mapping = parser.get_data()

import parser
all_imgs,classes_count,class_mapping = parser.get_data('VOCdevkit')

=======

all_imgs,classes_count,class_mapping = parser.get_data(sys.argv[1])
>>>>>>> refs/remotes/origin/master

with open('classes.json', 'w') as class_data_json:
    json.dump(class_mapping, class_data_json)

inv_map = {v: k for k, v in class_mapping.iteritems()}

pprint.pprint(classes_count)

random.shuffle(all_imgs)

num_imgs = len(all_imgs)

train_imgs = [s for s in all_imgs if s['imageset'] == 'trainval']
val_imgs = [s for s in all_imgs if s['imageset'] == 'test']

print('Num train samples {}'.format(len(train_imgs)))
print('Num val samples {}'.format(len(val_imgs)))

import data_generators

data_gen_train = data_generators.get_anchor_gt(train_imgs,class_mapping,classes_count,C)
data_gen_val = data_generators.get_anchor_gt(val_imgs,class_mapping,classes_count,C)


import resnet
from keras import backend as K
from keras.optimizers import Adam, SGD
from keras.layers import Input
from keras.models import Model
import losses

if K.image_dim_ordering() == 'th':
	input_shape_img = (3, None, None)
else:
	input_shape_img = (None, None, 3)

img_input = Input(shape=input_shape_img)

roi_input = Input(shape=(C.num_rois, 4))

# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = resnet.resnet_base(img_input)

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn = resnet.rpn(shared_layers,num_anchors)

# the classifier is build on top of the base layers + the ROI pooling layer + extra layers
classifier = resnet.classifier(shared_layers,roi_input,C.num_rois,nb_classes=len(classes_count)+1)

# define the full model
model = Model([img_input,roi_input],rpn + [classifier])

try:
	if K.image_dim_ordering() == 'th'		:
<<<<<<< HEAD
		hdf5_filepath = 'resnet50_weights_th_dim_ordering_th_kernels.h5'
=======
		hdf5_filepath = 'resnet50_weights_th_dim_ordering_th_kernels_notop.h5'
>>>>>>> refs/remotes/origin/master
	else:
		hdf5_filepath = 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

	model.load_weights(hdf5_filepath, by_name=True)
except:
	print('Could not load pretrained model weights')



<<<<<<< HEAD
#hdf5_filepath = 'model_frcnn.regr.no_bn.hdf5hdf5'
print 'loading weights from ', hdf5_filepath
resnet.load_weights_from_hdf5_group_by_name(model,hdf5_filepath)
#model.load_weights('./model_frcnn.regr.2.hdf5')
optimizer = Adam(lr = 1e-5)
=======
optimizer = Adam(1e-5)
>>>>>>> refs/remotes/origin/master

model.compile(optimizer=optimizer, loss=[losses.rpn_loss, losses.robust_l1_loss, 'categorical_crossentropy'])

nb_epoch = 1000

best_val_loss = 1e9
nb_epochs = 50
avg_loss_rpn = []
avg_loss_class = []

print 'starting training'

for i in range(1,len(train_imgs) * nb_epochs + 1):

	if i%2000 == 0:

		# run validation
		val_rpn_loss = 0
		val_class_losses = 0

		num_samples_for_val = 1000
		for j in range(num_samples_for_val):

			(X1,Y1_class,Y1_regr,X2,Y2) = data_gen_val.next()
			loss_total,loss_rpn_class,loss_rpn_regr,loss_class = model.test_on_batch([X1,X2],[Y1_class,Y1_regr,Y2])
			val_class_losses += loss_class
			val_rpn_loss += loss_rpn_class + loss_rpn_regr

		val_rpn_loss = val_rpn_loss / float(num_samples_for_val)
		val_class_losses = val_class_losses / float(num_samples_for_val)
		total_loss = val_rpn_loss + val_class_losses

		print('Validation losses = rpn: {}, classifier: {}'.format(val_rpn_loss,val_class_losses))

		if total_loss < best_val_loss:
			best_val_loss = total_loss
<<<<<<< HEAD
			model.save_weights('./model_frcnn.regr.no_bn.2.hdf5.hdf5')
=======
			model.save_weights('model_frcnn.hdf5')
>>>>>>> refs/remotes/origin/master

	(X1,Y1_class,Y1_regr,X2,Y2) = data_gen_train.next()

	loss_total,loss_rpn_class,loss_rpn_regr,loss_class = model.train_on_batch([X1,X2],[Y1_class,Y1_regr,Y2])

	avg_loss_rpn.append(loss_rpn_class + loss_rpn_regr)
	avg_loss_class.append(loss_class)

	if len(avg_loss_rpn) == 10:
		print('rpn,{},classifier,{}'.format(sum(avg_loss_rpn)/10.0,sum(avg_loss_class)/10.))
		avg_loss_rpn = []
		avg_loss_class = []
