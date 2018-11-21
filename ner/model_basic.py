import tensorflow as tf
import model_helper as _mh
import hparams_module as _hm

def learning_rate_update(hparams,global_step):
	return tf.train.exponential_decay(hparams.learning_rate,
								global_step,\
								hparams.decay_step,\
								hparams.decay_rate,\
								staircase=True)

def get_learning_rate(learning_rate, hidden_size, learning_rate_warmup_steps):
	with tf.name_scope("learning_rate"):
		warmup_steps = tf.to_float(learning_rate_warmup_steps)
		step = tf.to_float(tf.train.get_or_create_global_step())
		learning_rate *= (hidden_size ** -0.5)

		learning_rate *= tf.minimum(1.0, step / warmup_steps)

		learning_rate *= tf.rsqrt(tf.maximum(step, warmup_steps))

		return learning_rate
def get_lazy_opt(hparams):	
	learning_rate = get_learning_rate(
		learning_rate = hparams.lazy_learningrate,
		hidden_size = hparams.num_units,
		learning_rate_warmup_steps = hparams.learning_rate_warmup_steps)

	optimizer = tf.contrib.opt.LazyAdamOptimizer(
		learning_rate,
		beta1 = hparams.optimizer_adam_beta1,
		beta2 = hparams.optimizer_adam_beta2,
		epsilon = hparams.optimizer_adam_epsilon)
	return optimizer,learning_rate
def optimizer(hparams,loss,global_step):
	opt = None
	learning_rate = learning_rate_update(hparams,global_step)
	learning_rate = tf.maximum(tf.constant(0.00004),learning_rate)
	if hparams.opttype == 'SGD':
		opt = tf.train.GradientDescentOptimizer(learning_rate)
	if hparams.opttype == 'Adam':
		opt = tf.train.AdamOptimizer(learning_rate)
	if hparams.opttype == 'Nadam':
		opt = tf.contrib.opt.NadamOptimizer(learning_rate)
	if hparams.opttype == 'Lazy':
		opt,learning_rate = get_lazy_opt(hparams)

	gradient, vars_param = zip(*opt.compute_gradients(loss))
	clip_gradient,_ = _mh.gradient_clip(gradient)
	apply_gradient_op = opt.apply_gradients(zip(clip_gradient,vars_param),global_step=global_step)

	variable_averages = tf.train.ExponentialMovingAverage(0.9999,
                                                        global_step)
	variables_averages_op = variable_averages.apply(tf.trainable_variables())
	with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
		train_op = tf.no_op(name='train')
	
	return train_op,learning_rate
