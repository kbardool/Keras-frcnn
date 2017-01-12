from keras import backend as K

def rpn_loss(y_true, y_pred):
	return K.sum(y_true[:,:9,:,:] * K.binary_crossentropy(y_pred[:,:,:,:], y_true[:,9:,:,:]))/256.0

def robust_l1_loss(y_true, y_pred):
	x = y_true - y_pred
	x_abs = K.abs(x)
	x_bool = K.lesser_equal(x_abs,1.0)
	return  K.mean(x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))
