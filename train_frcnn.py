import random
import pprint
import sys
import json
import config
import pascal_voc_parser as parser

sys.setrecursionlimit(40000)

C = config.Config()
C.num_rois = 8


all_imgs,classes_count,class_mapping = parser.get_data(sys.argv[1])

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

data_gen_train.next()

import resnet
from keras import backend as K
from keras.optimizers import Adam, SGD
from keras.layers import Input
from keras.callbacks import  ModelCheckpoint
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
classifier = resnet.classifier(shared_layers, roi_input, C.num_rois, nb_classes=len(classes_count)+1)

# define the full model
model = Model([img_input, roi_input], rpn + classifier)

try:
	if K.image_dim_ordering() == 'th'		:
		hdf5_filepath = 'resnet50_weights_th_dim_ordering_th_kernels_notop.h5'
	else:
		hdf5_filepath = 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

	model.load_weights(hdf5_filepath, by_name=True)
except:
	print('Could not load pretrained model weights')



optimizer = Adam(1e-5)
model.compile(optimizer=optimizer, loss=[losses.rpn_loss_cls, losses.rpn_loss_regr, losses.class_loss_cls, losses.class_loss_regr])

model.summary()
nb_epoch = 1000

best_val_loss = 1e9
nb_epochs = 50
avg_loss_rpn = []
avg_loss_class = []
model_checkpoint = ModelCheckpoint('model_frcnn.hdf5',monitor='val_loss',verbose=1,save_best_only=True,save_weights_only=True)
model.fit_generator(data_gen_train,samples_per_epoch=2000,nb_epoch=100,callbacks=[model_checkpoint],
					validation_data=data_gen_val,nb_val_samples=1000)
