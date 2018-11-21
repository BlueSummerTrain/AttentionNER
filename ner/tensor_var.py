import tensorflow as tf

def _variable(name, shape, initializer):
  dtype = tf.float32
  var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var

def _variable_with_weight_decay(name, shape, stddev, l2_loss_rate,initializer=None):
  dtype = tf.float32
  if initializer == None:
    initializer = tf.truncated_normal_initializer(stddev=stddev, dtype=dtype)
    var = _variable(name, shape,initializer)
  else:
    var = _variable(name, shape,initializer)
  if l2_loss_rate is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), l2_loss_rate, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var