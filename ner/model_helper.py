# encoding=utf-8
import tensorflow as tf
import time
import os
from rlog import _log_normal, _log_warning, _log_info, _log_error

from hparams_module import MODE_TRAIN, MODE_INFER
from hparams_module import INITIALIZER_TYPE_UNIFORM, INITIALIZER_TYPE_GLOROT_NORMAL, INITIALIZER_TYPE_GLOROT_UNIFORM, INITIALIZER_TYPE_RANDOM_NORMAL
from hparams_module import RNN_UNIT_TYPE_LSTM, RNN_UNIT_TYPE_GRU, RNN_UNIT_TYPE_LAYER_NORM_LSTM
import hparams_module as _hm
from shutil import copy
from tensorflow.contrib.model_pruning.python import layers as pruning_layers
'''
some activation function define here
'''
def lrelu(x, leak=0.2, name='lrelu'):
    return tf.maximum(x, leak * x)

def ac_swish(x):
    '''
    New activation function from <Google Brain>.
    Usually be used for replacing old activation function 'relu'.
    '''
    return (x * tf.nn.sigmoid(x))
def elu(x,name='elu'):
    return tf.nn.elu(x)
def tanh(x,name='tanh'):
    return tf.nn.tanh(x)
def double_scale_tanh(x, scale1=10.0, scale2=1.0):
    return scale2 * tf.tanh(x / scale1)

'''
initialize tensor variables method
'''
def get_initializer(itype=None, seed=None, init_weight=0.01):

    if (itype == INITIALIZER_TYPE_UNIFORM):
        if (init_weight is None):
            _log_error('Invalid init_weight(model_helper.get_global_initializer).')
            return None
        return tf.random_uniform_initializer(-init_weight, init_weight, seed=seed)
    elif (itype == INITIALIZER_TYPE_GLOROT_NORMAL):
        return tf.contrib.keras.initializers.glorot_normal(seed=seed)
    elif (itype == INITIALIZER_TYPE_GLOROT_UNIFORM):
        return tf.contrib.keras.initializers.glorot_uniform(seed=seed)
    elif (itype == INITIALIZER_TYPE_RANDOM_NORMAL):
        if (init_weight is None):
            _log_error('Invalid init_weight(model_helper.get_global_initializer).')
            return None
        return tf.random_normal_initializer(mean=0.0, stddev=init_weight, seed=seed, dtype=tf.float32)

    _log_error('Invalid initializer type(model_helper.get_global_initializer).')
    return None
        
def create_single_cell_RNN(unit_type, num_units, dropout, mode, residual=False, forget_bias=1.0, seq=0):
    '''
    Create single neuron cell for RNN structure.

    @param mode
    @param unit_type: 3 types for choosing, [lstm, gru, layer_norm_lstm]
    @param forget_bias: add forget_bias (default: 1) to the biases of the
                        forget gate in order to reduce the scale of
                        forgetting in the beginning of the training.
    '''

    _dropout = 0.0
    if (mode == MODE_TRAIN and dropout < 1.0 and dropout > 0.0):
        _dropout = dropout

    single_cell = None
    if _hm.NEED_PRUNING:
        unit_type = 'Masked_LSTM'
    _log_normal('Create single RNN cell with type:%s, units:%d, id=%d, dropout:%2f, residual:%s' % (unit_type, num_units, seq, dropout, ('true' if (residual) else 'false')))

    ac = tanh

    if _hm.NEED_PRUNING:
        single_cell = pruning_layers.rnn_cells.MaskedBasicLSTMCell(num_units, forget_bias=forget_bias, activation=ac)
    else:
        if (unit_type == RNN_UNIT_TYPE_LSTM):
            single_cell = tf.contrib.rnn.BasicLSTMCell(num_units, forget_bias=forget_bias, activation=ac)
        elif (unit_type == RNN_UNIT_TYPE_GRU):
            single_cell = tf.contrib.rnn.GRUCell(num_units, activation=ac)
        elif (unit_type == RNN_UNIT_TYPE_LAYER_NORM_LSTM):
            single_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units, forget_bias=forget_bias, layer_norm=True, activation=ac)

    if (_dropout > 0.0):
        single_cell = tf.contrib.rnn.DropoutWrapper(cell=single_cell, input_keep_prob=(1.0 - _dropout))

    if (residual):
        single_cell = tf.contrib.rnn.ResidualWrapper(single_cell)

    return single_cell


def create_cell_list_for_RNN(unit_type, num_units, num_layers, dropout, mode, forget_bias=1.0, need_wrap=True):
    '''
    Create MultiRNNCell or single-cell if num_layers==1.

    @param num_layers: number of layers.
    @param mode: [train, infer].string.
    @param unit_type: 3 types for choosing, [lstm, gru, layer_norm_lstm]
    @param forget_bias: add forget_bias (default: 1) to the biases of the
                        forget gate in order to reduce the scale of
                        forgetting in the beginning of the training.
    '''
    cell_list = []

    for i in range(num_layers):
        single_cell = create_single_cell_RNN(unit_type, num_units, dropout, mode,
                                             residual=(i >=1),
                                             forget_bias=forget_bias, seq=i + 1)
        cell_list.append(single_cell)


    if (need_wrap):
        if (len(cell_list) == 1):
            # for single layer
            return cell_list[0]
        else:
            return  tf.contrib.rnn.MultiRNNCell(cell_list)
    else:
        return cell_list


def create_bidirectional_RNN(unit_type=None, num_units=None, num_layers=None, dropout=0.2, mode='', time_major=True, forget_bias=1.0,\
                             inputs=None, sequence_length=None, dtype=tf.float32):

    fw_cell = create_cell_list_for_RNN(unit_type, num_units, num_layers, dropout, mode, forget_bias)
    bw_cell = create_cell_list_for_RNN(unit_type, num_units, num_layers, dropout, mode, forget_bias)

    bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, inputs, dtype=dtype, sequence_length=sequence_length, time_major=time_major)

    return tf.concat(bi_outputs, -1), bi_state



def gradient_clip(gradients, max_gradient_norm=5.0):
    '''
    Gradients by loss, params(tf.trainable_variables()), colocate_gradients_with_ops=True/False.
    Used for opt.apply_gradients.
          e.g. opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)
    @param max_gradient_norm: default 5.0.

    @return clipped_gradients & gradient_norm_summary(for record).
    '''
    clipped_gradients, gradient_norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
    gradient_norm_summary = [tf.summary.scalar("grad_norm", gradient_norm)]
    gradient_norm_summary.append(tf.summary.scalar("clipped_gradient", tf.global_norm(clipped_gradients)))

    return clipped_gradients, gradient_norm_summary


def global_init(sess=None, only_table=False):
    '''
    Execute tf[global_variables_initializer, local_variables_initializer, tables_initializer].

    param: only_table, Should init table variables only by loading model.
    '''
    if (sess is None):
        _log_error('Invalid TF Session.')
        return

    init = tf.global_variables_initializer()
    local_init = tf.local_variables_initializer()
    table_init = tf.tables_initializer()
    if (only_table):
        sess.run([table_init])
        _log_info('table variables have been inited...')
    else:
        sess.run([init, local_init, table_init])
        _log_info('global & local & table variables have been inited...')

def load_model(saver, ckpt, session):
    start_time = time.time()
    saver.restore(session, ckpt)
    _log_info("loaded model parameters from %s, time %.2fs" % (ckpt, time.time() - start_time))


def create_or_load_model(model_dir, session,saver):
    ckpt = tf.train.get_checkpoint_state(model_dir)
    step_dir = None 
    if ckpt and ckpt.model_checkpoint_path:
        load_model(saver, ckpt.model_checkpoint_path, session)
        global_init(session, True)
        step_dir = ckpt.model_checkpoint_path
    else:
        global_init(session)
        _log_info('Create new model...')
    return step_dir

def save_bestdir_model(save_dir,global_step):
    filelist = os.listdir(save_dir)
    if len(filelist):
        for i in filelist:
            c_path = os.path.join(save_dir,i)
            os.remove(c_path)

    filename0 = 'model.ckpt-'+str(global_step)+'.data-00000-of-00001'
    filename1 = 'model.ckpt-'+str(global_step)+'.index'
    filename2 = 'model.ckpt-'+str(global_step)+'.meta'
    filename3 = 'checkpoint'
    filename4 = 'ner.pbtxt'
    copy(save_dir+'../'+filename0,save_dir)
    copy(save_dir+'../'+filename1,save_dir)
    copy(save_dir+'../'+filename2,save_dir)
    copy(save_dir+'../'+filename3,save_dir)
    copy(save_dir+'../'+filename4,save_dir)
# attention related
def create_attention_mechanism(attention_mode='normed_bahdanau', num_units=None, memory=None, source_sequence_length=None):
    if (num_units == None or memory == None or source_sequence_length == None):
        _log_error('Invalid params[create_attention_mechanism].')
        return None
    attention_mechanism = None
    if (attention_mode == _hm.RNN_ATTENTION_LUONG):
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units, memory, memory_sequence_length=source_sequence_length)
    elif (attention_mode == _hm.RNN_ATTENTION_SLUONG):
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units, memory, memory_sequence_length=source_sequence_length, scale=True)
    elif (attention_mode == _hm.RNN_ATTENTION_BAHDANAU):
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units, memory, memory_sequence_length=source_sequence_length)
    elif (attention_mode == _hm.RNN_ATTENTION_NBAHDANAU):
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units, memory, memory_sequence_length=source_sequence_length, normalize=True)
    else:
        _log_error('Unknown attention mode:%s' % attention_mode)
        return None

    return attention_mechanism

# not necessary, for creating picture
def _create_attention_images_summary(final_context_state):
    attention_images = (final_context_state.alignment_history.stack())
    attention_images = tf.expand_dims(tf.transpose(attention_images, [1, 2, 0]), -1)
    attention_images *= 255
    attention_summary = tf.summary.image("attention_images", attention_images)
    return attention_summary
def _get_max_time(tensor):
    time_axis = 0 if self.time_major else 1
    return tensor.shape[time_axis].value or tf.shape(tensor)[time_axis]