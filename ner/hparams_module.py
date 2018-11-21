# encoding=utf-8
# constants of training mode.
MODE_TRAIN = 'train'
MODE_INFER = 'infer'
# constants of initializer type
INITIALIZER_TYPE_UNIFORM = 'uniform'
INITIALIZER_TYPE_GLOROT_NORMAL = 'glorot_normal'
INITIALIZER_TYPE_GLOROT_UNIFORM = 'glorot_uniform'
INITIALIZER_TYPE_RANDOM_NORMAL = 'random normal'
# constants of attention mode
RNN_ATTENTION_LUONG = 'luong'
RNN_ATTENTION_SLUONG = 'scaled_luong'
RNN_ATTENTION_BAHDANAU = 'bahdanau'
RNN_ATTENTION_NBAHDANAU = 'normed_bahdanau'
# constants of unit type
RNN_UNIT_TYPE_LSTM = 'lstm'
RNN_UNIT_TYPE_GRU = 'gru'
RNN_UNIT_TYPE_LAYER_NORM_LSTM = 'layer_norm_lstm'
########data relate############
MAIN_PATH = '../ner_train_data/ner_train/'
OBJ_PATH = '../ner_train_data/ner_train_obj/'
WORD_VOC = 'vocab.na.data'
NER_LABEL = 'vocab.lf.data'
#########model purning parameter####
NEED_PRUNING = False
PRUNING_PARAMS='name=ner_pruning,\
                begin_pruning_step=0,\
                end_pruning_step=-1,\
                target_sparsity=0.9,\
                pruning_frequency=10,\
                sparsity_function_begin_step=4000,\
                sparsity_function_end_step=400000'
# ##################################
# ==== rnn and cnn releated ========
# ##################################
class _ParamsNerRnnModel:
    def __init__(self):
        self.learning_rate = 0.002
        self.num_layers = 4
        self.decoder_layer_num_more = 0
        self.num_units = 256
        self.residual = True
        self.time_major = True
        self.src_vocab_size = 6180
        self.tgt_vocab_size = 16
        self.embedding_size = 128
        self.unit_type = RNN_UNIT_TYPE_GRU
        self.attention_mode = RNN_ATTENTION_SLUONG
        self.dropout = 0.15
        self.forget_bias = 1.0
        self.share_vocab = False
        self.out_dir = './models/'
        self.train_steps = 400000 #train model total steps
        self.train_save_steps = 100 #saving model steps 
        self.opttype = 'Adam'
        self.use_l2_loss = True		#use l2 loss
        self.l2_rate = 0.0001
        self.train_type = 'main'	#if 'main',train all Name entity,if 'obj',train object entity
        self.batch_size = 1024
        self.decay_step = 100
        self.decay_rate = 0.996
        self.saver_max_time = 10
        self.read_data_len = 10000	#read data len from data file per time
        self.add_noise = False
        self.lazy_learningrate = 2.0
        self.learning_rate_warmup_steps = 16000
        self.optimizer_adam_beta1 = 0.9
        self.optimizer_adam_beta2 = 0.997
        self.optimizer_adam_epsilon = 1e-09
ner_mode_params = _ParamsNerRnnModel()
