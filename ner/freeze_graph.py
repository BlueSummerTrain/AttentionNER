import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.python.framework import meta_graph
dir(tf.contrib)

saved_graph_name = './models/best/ner.pbtxt'
saved_ckpt_name = './models/best/model.ckpt-11926'
input_node_name = 'encoder_input_data:0'
out_node_name = 'NER_output'

output_frozen_graph_name = '../ner_corpus/ner.pb'

freeze_graph.freeze_graph(input_graph=saved_graph_name,\
						input_saver='', \
						input_binary=False, \
						input_checkpoint=saved_ckpt_name, \
						output_node_names=out_node_name, \
						restore_op_name='', filename_tensor_name='', \
						output_graph=output_frozen_graph_name, \
						clear_devices=True, initializer_nodes='')

# optimized model is not ready, there's still bug in the function.
'''
input_graph_def = tf.GraphDef()
with tf.gfile.Open(output_frozen_graph_name, 'r') as f:
    data = f.read()
    input_graph_def.ParseFromString(data)

output_graph_def = optimize_for_inference_lib.optimize_for_inference(input_graph_def, [], [out_node_name], tf.string.as_datatype_enum)

f = tf.gfile.FastGFile(optimized_model_name, 'w')
f.write(output_graph_def.SerializeToString())
'''
