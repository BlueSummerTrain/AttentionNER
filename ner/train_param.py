import tensorflow as tf
import numpy as np
import re

def count_network_param(train_var):
	param_size = lambda v: reduce(lambda x,y: x*y, v.get_shape().as_list())
	return sum(param_size(v) for v in train_var)