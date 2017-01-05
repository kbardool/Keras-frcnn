from __future__ import print_function
from __future__ import absolute_import

from keras.layers import merge, Input
from keras.layers import Dense, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D, TimeDistributed
from keras.layers import BatchNormalization
from keras.models import Model
from keras import backend as K
from RoiPoolingConv import RoiPoolingConv
import numpy as np
import pdb
import h5py

bn_mode = 2

def load_weights_from_hdf5_group_by_name(model, hdf5_filepath):
    f = h5py.File(hdf5_filepath)
    ''' Name-based weight loading
    (instead of topological weight loading).
    Layers that have no matching name are skipped.
    '''
    if hasattr(model, 'flattened_layers'):
        # support for legacy Sequential/Merge behavior
        flattened_layers = model.flattened_layers
    else:
        flattened_layers = model.layers

    if 'nb_layers' in f.attrs:
        raise Exception('The weight file you are trying to load is' +
                        ' in a legacy format that does not support' +
                        ' name-based weight loading.')
    else:
        # new file format
        layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]

        # Reverse index of layer name to list of layers with name.
        index = {}
        for layer in flattened_layers:
            if layer.name:
                index.setdefault(layer.name, []).append(layer)

        # we batch weight value assignments in a single backend call
        # which provides a speedup in TensorFlow.
        weight_value_tuples = []
        num_valid_layers = 0
        for k, name in enumerate(layer_names):
            g = f[name]
            weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
            weight_values = [g[weight_name] for weight_name in weight_names]

            for layer in index.get(name, []):
                symbolic_weights = layer.weights
                if len(weight_values) != len(symbolic_weights):
                    raise Exception('Layer #' + str(k) +
                                    ' (named "' + layer.name +
                                    '") expects ' +
                                    str(len(symbolic_weights)) +
                                    ' weight(s), but the saved weights' +
                                    ' have ' + str(len(weight_values)) +
                                    ' element(s).')
                else:
                    num_valid_layers += 1
                # set values
                for i in range(len(weight_values)):
                    #print('setting val for layer {}'.format(name))
                    weight_value_tuples.append(
                        (symbolic_weights[i], weight_values[i]))
        print('Loaded {} layers by name'.format(num_valid_layers))
        K.batch_set_value(weight_value_tuples)


def identity_block(input_tensor, kernel_size, filters, stage, block):
    '''The identity_block is the block that has no conv layer at shortcut
    # Arguments
            input_tensor: input tensor
            kernel_size: defualt 3, the kernel size of middle conv layer at main path
            filters: list of integers, the nb_filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names
    '''
    nb_filter1, nb_filter2, nb_filter3 = filters

    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1
    
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, 1, 1, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, mode=bn_mode, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, kernel_size, kernel_size, border_mode='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, mode=bn_mode, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter3, 1, 1, name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, mode=bn_mode, name=bn_name_base + '2c')(x)

    x = merge([x, input_tensor], mode='sum')
    x = Activation('relu')(x)
    return x


def identity_block_td(input_tensor, kernel_size, filters, stage, block):
    '''The identity_block_td is the block that has no conv layer at shortcut
    # Arguments
            input_tensor: input tensor
            kernel_size: defualt 3, the kernel size of middle conv layer at main path
            filters: list of integers, the nb_filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names
    '''
    nb_filter1, nb_filter2, nb_filter3 = filters

    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = TimeDistributed(Convolution2D(nb_filter1, 1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis+1, mode=bn_mode, name=bn_name_base + '2a')(x)

    x = Activation('relu')(x)

    x = TimeDistributed(Convolution2D(nb_filter2, kernel_size, kernel_size, border_mode='same'), name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis+1, mode=bn_mode, name=bn_name_base + '2b')(x)

    x = Activation('relu')(x)

    x = TimeDistributed(Convolution2D(nb_filter3, 1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis+1, mode=bn_mode, name=bn_name_base + '2c')(x)

    x = merge([x, input_tensor], mode='sum')
    x = Activation('relu')(x)

    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    '''conv_block is the block that has a conv layer at shortcut
    # Arguments
            input_tensor: input tensor
            kernel_size: defualt 3, the kernel size of middle conv layer at main path
            filters: list of integers, the nb_filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    '''
    nb_filter1, nb_filter2, nb_filter3 = filters
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, 1, 1, subsample=strides, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, mode=bn_mode, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, kernel_size, kernel_size, border_mode='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, mode=bn_mode, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter3, 1, 1, name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, mode=bn_mode, name=bn_name_base + '2c')(x)

    shortcut = Convolution2D(nb_filter3, 1, 1, subsample=strides, name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, mode=bn_mode, name=bn_name_base + '1')(shortcut)

    x = merge([x, shortcut], mode='sum')
    x = Activation('relu')(x)
    return x

def conv_block_td(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    '''conv_block_td is the block that has a conv layer at shortcut
    # Arguments
            input_tensor: input tensor
            kernel_size: defualt 3, the kernel size of middle conv layer at main path
            filters: list of integers, the nb_filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    '''
    nb_filter1, nb_filter2, nb_filter3 = filters
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = TimeDistributed(Convolution2D(nb_filter1, 1, 1, subsample=strides), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis+1, mode=bn_mode, name=bn_name_base + '2a')(x)

    x = Activation('relu')(x)

    x = TimeDistributed(Convolution2D(nb_filter2, kernel_size, kernel_size, border_mode='same'), name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis+1, mode=bn_mode, name=bn_name_base + '2b')(x)

    x = Activation('relu')(x)

    x = TimeDistributed(Convolution2D(nb_filter3, 1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis+1, mode=bn_mode, name=bn_name_base + '2c')(x)


    shortcut = TimeDistributed(Convolution2D(nb_filter3, 1, 1, subsample=strides), name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis+1, mode=bn_mode, name=bn_name_base + '1')(shortcut)

    x = merge([x, shortcut], mode='sum')
    x = Activation('relu')(x)
    return x

def resnet_base(input_tensor=None):

    # Determine proper input shape
    if K.image_dim_ordering() == 'th':
        input_shape = (3, None, None)
    else:
        input_shape = (None, None, 3)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    x = ZeroPadding2D((3, 3))(img_input)
    x = Convolution2D(64, 7, 7, subsample=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, mode=bn_mode, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    return x

def classifier_layers(x):
    x = conv_block_td(x, 3, [512, 512, 2048], stage=5, block='a', strides=(1, 1))
    x = identity_block_td(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block_td(x, 3, [512, 512, 2048], stage=5, block='c')

    x = TimeDistributed(AveragePooling2D((7, 7)), name='avg_pool')(x)

    return x

def rpn(base_layers):

    x = Convolution2D(512, 3, 3, border_mode = 'same', activation='relu', init='normal')(base_layers)

    num_out_for_class = 9
    x_class = Convolution2D(num_out_for_class, 1, 1, activation='sigmoid', init='normal')(x)
    return (x_class)

def classifier(base_layers,input_rois,num_rois,nb_classes = 21):

    pooling_regions = 7

    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers,input_rois])
    out_class  = classifier_layers(out_roi_pool)
    out_class  = TimeDistributed(Flatten())(out_class)
    out_class  = TimeDistributed(Dense(nb_classes,activation='softmax'),name='dense_{}'.format(nb_classes))(out_class)

    return (out_class)