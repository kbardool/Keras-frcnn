from keras import backend as K

num_anchors = 9

lambda_rpn_class = 10.0
lambda_rpn_regr = 100.0


def rpn_loss(y_true, y_pred):
	return lambda_rpn_class * K.sum(y_true[:, :num_anchors, :, :] * K.binary_crossentropy(y_pred[:, :, :, :],
																						  y_true[:, num_anchors:, :,
																						  :])) / 256.0


def robust_l1_loss(y_true, y_pred):
	x = y_true[:, 4 * num_anchors:, :, :] - y_pred
	x_abs = K.abs(x)
	x_bool = K.lesser_equal(x_abs, 1.0)
	return lambda_rpn_regr * K.sum(
		y_true[:, :4 * num_anchors, :, :] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / 256.
