# encoding=utf-8
import numpy as np
import sys
import tensorflow as tf
from tensorflow.python.layers import core as layers_core
from tensor_var import _variable_with_weight_decay
from train_param import count_network_param
from rlog import _log_normal, _log_warning, _log_info, _log_error

import model_helper as _mh
import model_basic as _mb
import hparams_module as _hm

from data_utils import NMT_UNK, NMT_SOS, NMT_EOS, NMT_UNK_ID

from tensorflow.contrib.model_pruning.python import pruning
from tensorflow.contrib.model_pruning.python import layers as pruing_layers

class NERRnnModel():

    def __init__(self, hparams=None, mode='', seed=None, init_weight=0.01,dtype=tf.float32):
        self.encoder_input_data = tf.placeholder(tf.int32, [None, None], name='encoder_input_data')
        self.decoder_output_data = tf.placeholder(tf.int32, [None, None], name='decoder_output_data')

        self.tgt_vocab_size = hparams.tgt_vocab_size

        len_temp = tf.sign(tf.add(tf.abs(tf.sign(self.encoder_input_data)),1))
        self.seq_length_encoder_input_data = tf.cast(tf.reduce_sum(len_temp,-1),tf.int32)
        self.batch_size = tf.size(self.seq_length_encoder_input_data)

        self.num_layers = hparams.num_layers
        self.decoder_layer_num_more = hparams.decoder_layer_num_more

        self.unit_type = hparams.unit_type
        self.num_units = hparams.num_units
        self.dropout = hparams.dropout
        self.forget_bias = hparams.forget_bias

        self.attention_mode = hparams.attention_mode

        self.time_major = hparams.time_major
        self.residual = hparams.residual
        self.train_type = hparams.train_type
        self.mode = mode
        #l2 loss relate
        self.use_l2_loss = hparams.use_l2_loss
        self.l2_rate = hparams.l2_rate

        self.embedding_size = hparams.embedding_size
        self.dtype = dtype
        tf.get_variable_scope().set_initializer(tf.contrib.keras.initializers.glorot_normal(seed=None))
        self.global_step = tf.train.get_or_create_global_step()
        #pruing paramter
        if _hm.NEED_PRUNING:
            pruning_hparams = pruning.get_pruning_hparams().parse(_hm.PRUNING_PARAMS)
            pruning_obj = pruning.Pruning(pruning_hparams, global_step=self.global_step)
        #embeding variable
        with tf.variable_scope('embedding_var') as scope:
            shape = [hparams.src_vocab_size, hparams.embedding_size]
            self.embedding_encoder = tf.Variable(tf.random_uniform(shape,-0.01, 0.01), dtype=tf.float32, name="embedding")
            self.embedding_decoder = self.embedding_encoder
        
        self.crf_transmit = tf.get_variable("crf_transmit",
                                            [self.tgt_vocab_size,self.tgt_vocab_size],
                                            initializer=tf.random_normal_initializer(0., 512 ** -0.5))
        res = self._build_graph()

        if (self.mode == _hm.MODE_TRAIN):
            self.loss = res[1]
            self.update,self.learning_rate = _mb.optimizer(hparams,self.loss,self.global_step)
            if _hm.NEED_PRUNING:
                self.mask_update_op = pruning_obj.conditional_mask_update_op()
                pruning_obj.add_pruning_summaries()
        #infer here
        else:
            logits = res[0]
            viterbi_sequence,_ =tf.contrib.crf.crf_decode(logits,\
                                                    self.crf_transmit,\
                                                    self.seq_length_encoder_input_data)
            self.neroutput = tf.identity(viterbi_sequence,name="NER_output")
        
        self.saver = tf.train.Saver(tf.global_variables(),max_to_keep = hparams.saver_max_time)
        self.merged_summary = tf.summary.merge_all()

    def _build_graph(self):
        _log_info('Begin to build graph...')
        loss= tf.constant(0.0)
        encoder_outputs, encoder_state = self._build_encoder()
        logits = self._build_decoder(encoder_outputs)
        if self.mode == _hm.MODE_TRAIN:
            loss = self._compute_loss(logits)
        return logits, loss

    def _compute_loss(self, logits):

        target_output = self.decoder_output_data
        log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(logits,\
                                                                            target_output,\
                                                                            self.seq_length_encoder_input_data,
                                                                            transition_params=self.crf_transmit)
        loss = tf.reduce_mean(-log_likelihood)
        loss = loss + tf.add_n(tf.get_collection('losses'), name='total_loss')
        return loss

    def _build_encoder(self):
        source = self.encoder_input_data

        if (self.time_major):
            source = tf.transpose(source)
        self.source_test = source
        encoder_emb_inp = tf.nn.embedding_lookup(self.embedding_encoder, source)

        num_bi_layers = int(self.num_layers / 2)
        _log_info('encoder_type:bi-lstm, num_bi_layers:%d.' % (num_bi_layers))

        encoder_outputs, bi_encoder_state = _mh.create_bidirectional_RNN(self.unit_type, self.num_units, num_bi_layers, self.dropout,\
                                                                             self.mode, self.time_major, self.forget_bias,\
                                                                             encoder_emb_inp, sequence_length=self.seq_length_encoder_input_data,\
                                                                             dtype=self.dtype)
        '''
        if num_bi_layers > 1:
            fw_c,fw_h = bi_encoder_state[0][-1]
            bw_c,bw_h = bi_encoder_state[1][-1]
        else:
            fw_c,fw_h = bi_encoder_state[0]
            bw_c,bw_h = bi_encoder_state[1]
        encoder_state = [fw_h,bw_h]
        '''
        encoder_state = []

        return encoder_outputs, encoder_state

    def _build_decoder(self, encoder_outputs):
        max_encoder_length = tf.reduce_max(self.seq_length_encoder_input_data)
        maximum_iterations = tf.to_int32(tf.round(tf.to_float(max_encoder_length)))
        '''
        build user defined helper
        '''
        def initial_fn():
            sos_time_slice = tf.ones([self.batch_size],dtype=tf.int32) * 2
            sos_step_embedded = tf.nn.embedding_lookup(self.embedding_decoder,sos_time_slice)
            initial_input = tf.concat((sos_step_embedded,encoder_outputs[0]),1)
            initial_elements_finished = (self.seq_length_encoder_input_data<=0)
            return initial_elements_finished,initial_input
        def sample_fn(time,outputs,state):
            prediction_id = tf.to_int32(tf.argmax(outputs,axis=1))
            return prediction_id
        def next_inputs_fn(time,outputs,state,sample_ids):
            pad_step_embedded = tf.zeros([self.batch_size,self.num_units*2+self.embedding_size],dtype=tf.float32)
            pred_embedding = tf.nn.embedding_lookup(self.embedding_decoder,sample_ids)
            next_input = tf.concat((pred_embedding,encoder_outputs[time]),1)
            elements_finished = (time > self.seq_length_encoder_input_data)
            all_finished = tf.reduce_all(elements_finished)
            next_input = tf.cond(all_finished,lambda:pad_step_embedded,lambda:next_input)
            next_state = state
            return elements_finished,next_input,next_state

        decoder_helper = tf.contrib.seq2seq.CustomHelper(initial_fn,sample_fn,next_inputs_fn)

        with tf.variable_scope("decoder") as decoder_scope:
            # build decode cell begin
            memory = None
            if (self.time_major):
                memory = tf.transpose(encoder_outputs, [1, 0, 2])
            else:
                memory = encoder_outputs

            batch_size = self.batch_size
            source_sequence_length = self.seq_length_encoder_input_data

            attention_mechanism = _mh.create_attention_mechanism(self.attention_mode, self.num_units, memory, source_sequence_length)

            cell = _mh.create_cell_list_for_RNN(self.unit_type, self.num_units,\
                                                self.num_layers + self.decoder_layer_num_more,\
                                                self.dropout,\
                                                self.mode,\
                                                self.forget_bias)

            cell = tf.contrib.seq2seq.AttentionWrapper(cell, attention_mechanism,\
                                                       attention_layer_size=self.num_units,\
                                                       alignment_history=False,\
                                                       name='attention')

            decoder_initial_state = cell.zero_state(batch_size, self.dtype)
            # build decode cell end

            logits, sample_id, final_context_state = None, None, None
            l2_loss = 0.0
            '''
            train and infer used new helper
            '''
            decoder = tf.contrib.seq2seq.BasicDecoder(cell=cell,\
                                                    helper=decoder_helper,\
                                                    initial_state=decoder_initial_state)
            outputs, final_context_state, _ = \
            tf.contrib.seq2seq.dynamic_decode(decoder=decoder,\
                                              output_time_major=self.time_major,\
                                              impute_finished=True,\
                                              maximum_iterations=maximum_iterations)

            l2_reg = tf.contrib.layers.l2_regularizer(self.l2_rate)
            if _hm.NEED_PRUNING:
                output_layer = pruing_layers.core_layers.MaskedFullyConnected(self.tgt_vocab_size,
                                                            use_bias=True,
                                                            kernel_regularizer = l2_reg,
                                                            bias_regularizer = l2_reg,
                                                            name="output_projection")
            else:
                output_layer = layers_core.Dense(self.tgt_vocab_size,
                                                use_bias=True,
                                                kernel_regularizer = l2_reg,
                                                bias_regularizer = l2_reg,
                                                name="output_projection")
            logits = output_layer(outputs.rnn_output)
            l2_losses = tf.losses.get_regularization_losses()
            tf.add_to_collection('losses',tf.add_n(l2_losses)/tf.cast(self.batch_size,dtype=tf.float32))
            
            if self.time_major:
                logits = tf.transpose(logits, [1, 0, 2])
            return logits

    def train(self, sess, realv):
        assert self.mode == _hm.MODE_TRAIN
        feed = {self.encoder_input_data:realv[0],self.decoder_output_data:realv[1]}
        res = sess.run([self.update,\
                        self.loss,\
                        self.global_step,\
                        self.learning_rate,\
                        self.merged_summary], feed_dict=feed)

        #update pruning mask
        if _hm.NEED_PRUNING:
            sess.run(self.mask_update_op)
        return  res[1], res[2], res[4]

    def infer(self, sess, realv):
        feed = {self.encoder_input_data:realv}
        return sess.run(self.neroutput, feed_dict=feed)

