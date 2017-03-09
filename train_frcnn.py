import random
import pprint
import sys
import json
from keras_frcnn import config

sys.setrecursionlimit(40000)

C = config.Config()
C.num_rois = 2



from keras_frcnn.pascal_voc_parser import get_data

all_imgs,classes_count,class_mapping = get_data(sys.argv[1])


if 'bg' not in classes_count:
	classes_count['bg'] = 0
	class_mapping['bg'] = len(class_mapping)

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


from keras_frcnn import data_generators
from keras import backend as K

data_gen_train = data_generators.get_anchor_gt(train_imgs, class_mapping, classes_count,C, K.image_dim_ordering(), mode='train')
data_gen_val = data_generators.get_anchor_gt(val_imgs, class_mapping, classes_count,C, K.image_dim_ordering(), mode='train')

from keras_frcnn import resnet as nn
from keras.optimizers import Adam, SGD
from keras.layers import Input
from keras.callbacks import  ModelCheckpoint
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras_frcnn import losses

if K.image_dim_ordering() == 'th':
	input_shape_img = (3, None, None)
else:
	input_shape_img = (None, None, 3)

img_input = Input(shape=input_shape_img)

roi_input = Input(shape=(C.num_rois, 4))

# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input,trainable=True)

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn = nn.rpn(shared_layers,num_anchors)

# the classifier is build on top of the base layers + the ROI pooling layer + extra layers
classifier = nn.classifier(shared_layers, roi_input, C.num_rois, nb_classes=len(classes_count),trainable=True)

# define the full model
model = Model([img_input, roi_input], rpn + classifier)

try:
	print 'loading weights from ', C.base_net_weights
	model.load_weights(C.base_net_weights, by_name=True)
except:
	print('Could not load pretrained model weights. Weights can be found at {} and {}'.format(
		'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_th_dim_ordering_th_kernels_notop.h5',
        'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
        ))

optimizer = Adam(1e-6)
model.compile(optimizer=optimizer, loss=[losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors), losses.class_loss_cls, losses.class_loss_regr(C.num_rois,len(classes_count)-1)])

nb_epochs = 50

callbacks = [EarlyStopping(monitor='val_loss', patience=20, verbose=0),
				ModelCheckpoint(C.model_path, monitor='val_loss', save_best_only=True, verbose=0)]
train_samples_per_epoch = 2000 #len(train_imgs)
nb_val_samples = 500 # len(val_imgs),

print 'Starting training'

model.fit_generator(data_gen_train, samples_per_epoch=train_samples_per_epoch, nb_epoch= nb_epochs, validation_data=data_gen_val, nb_val_samples=nb_val_samples, callbacks=callbacks, max_q_size=10, nb_worker=1)

