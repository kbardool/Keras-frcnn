from keras import backend as K


def rpn_loss(y_true, y_pred):
	return K.sum(y_true[:,:9,:,:] * K.binary_crossentropy(y_pred[:,:,:,:], y_true[:,9:,:,:]))/256.0
