# encoding=utf-8
import os
import time
import codecs as _co
import argparse as _ap

import numpy as np
import tensorflow as tf
from rlog import _log_normal, _log_warning, _log_info, _log_error
import model_helper as _mh
import hparams_module as _hm
import tradition_data_utils as _tdu
from model_structure import NERRnnModel
from check_result import check_r
import create_data as _cd

from tqdm import tqdm
import math

#loading hparams
current_params = _hm.ner_mode_params
#root path
DATA_PATH = '../ner_corpus'
#check ner path
LABEL_FILE_INFER = os.path.join(DATA_PATH, 'test.lf.data')
NER_FILE_INFER = os.path.join(DATA_PATH, 'test.infer.ner')
if current_params.train_type == 'main':
    LABEL_FILE_INFER = LABEL_FILE_INFER + '_main'
    NER_FILE_INFER = NER_FILE_INFER + '_main'
else:
    LABEL_FILE_INFER = LABEL_FILE_INFER + '_obj'
    NER_FILE_INFER = NER_FILE_INFER + '_obj'
# vocabulary files.
DATA_VOCAB_FILE = os.path.join(DATA_PATH, _hm.WORD_VOC)
LABEL_VOCAB_FILE = os.path.join(DATA_PATH, _hm.NER_LABEL)
# recode result
INFER_PRECISION = './models/infer_precision'
LOSS_RECORD = './models/loss_record'

def train(gpu_conf):
    # define some varibiles maybe used
    epoch_avg_loss = []
    epoch = 0
    start_data_pos = 0
    infer_acc = 0.0
    global_step = 0
    #check path
    if not os.path.exists(current_params.out_dir):
        os.makedirs(current_params.out_dir)
    checkpoint_dir = os.path.join(current_params.out_dir,'model.ckpt')
    best_model_path = current_params.out_dir+'best/'
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)
    # begin training
    with tf.Graph().as_default() as g:
        with tf.Session(graph=g,config=tf.ConfigProto(gpu_options=gpu_conf)) as sess:

            model_train = NERRnnModel(hparams=current_params,mode=_hm.MODE_TRAIN,init_weight=0.01)
            latest_ckpt = _mh.create_or_load_model(current_params.out_dir, sess, model_train.saver)
            #log saver
            writer = tf.summary.FileWriter(current_params.out_dir, sess.graph)

            while global_step < current_params.train_steps:
                data_seq, labels, data_seq_small, labels_small, seq_lens, count_num = \
                _cd.train_data(current_params.train_type,start_data_pos)
                #merge data
                data_seq = data_seq + data_seq_small
                labels = labels + labels_small
                _log_info("current read data len:%d"%(len(data_seq)))

                if seq_lens <= (current_params.read_data_len * count_num)/2:
                    start_data_pos = 0
                else:
                    start_data_pos = start_data_pos + 1
                FILE_LENGTH = _tdu.init_dataset(data_seq, labels, mode='train')
                batch_size = current_params.batch_size
                batch_step = int(math.ceil(float(FILE_LENGTH) / float(batch_size)))
                step = 0

                tq = tqdm(range(batch_step))

                for itq in tq:
                    realv = _tdu.get_batch_data_for_train(batch_size = batch_size)
                    if len(realv) == 0:
                        break
                    (step_loss,global_step,logger_summary) = model_train.train(sess, realv)
                    epoch_avg_loss.append(step_loss)
                    tq.set_description('loss:%f, step:%d, epoch:%d, lrate:%f' % (step_loss, global_step, epoch, model_train.learning_rate.eval(session=sess)))

                    if (global_step > 0 and global_step % (current_params.train_save_steps) == 0):
                        #save loss record
                        _log_warning('epoch avg loss: %f' %(np.mean(epoch_avg_loss)))
                        with _co.open(LOSS_RECORD, 'a+', encoding='utf-8') as loss_w:
                            loss_w.write(str(global_step) + ' : ' + str(np.mean(epoch_avg_loss)) + '\n')
                        while (len(epoch_avg_loss) > 0):
                            epoch_avg_loss.remove(epoch_avg_loss[0])
                        #save models
                        model_train.saver.save(sess,checkpoint_dir,global_step=global_step)
                        infter_test = inference(gpu_conf)
                        #save best models
                        if infter_test >= infer_acc:
                           infer_acc = infter_test
                           _mh.save_bestdir_model(best_model_path,global_step)
                           _log_warning('saving model on step:%d,infer_acc:%f' %(global_step,infer_acc))
                        writer.add_summary(logger_summary, step)#save log
                    step = step + 1 # data step add
                tq.refresh()
                epoch = epoch + 1 #read data add

            #save the last time models
            model_train.saver.save(sess,checkpoint_dir,global_step=global_step)
            infter_test = inference(gpu_conf)

def inference(gpu_conf):
    #load infer data
    data_seq, labels = _cd.test_data(current_params.train_type)
    _tdu.init_dataset(data_seq, labels, mode='infer')
    #inference must build new session
    with tf.Graph().as_default() as g:
        with tf.Session(graph=g,config=tf.ConfigProto(gpu_options=gpu_conf)) as sess:

            model_infer = NERRnnModel(hparams=current_params, mode=_hm.MODE_INFER, init_weight=0.01)
            latest_ckpt = _mh.create_or_load_model(current_params.out_dir, sess, model_infer.saver)
            # create infer output file firstly.
            _co.open(NER_FILE_INFER, 'w', 'utf-8').write('')
            with _co.getwriter('utf-8')(tf.gfile.GFile(NER_FILE_INFER, mode='a')) as infer_f:
                # write empty char to make sure file exists.
                infer_f.write('')
                final_res = ''
                batch_size_infer = current_params.batch_size
                length = int(math.ceil(float(len(_tdu.src_datas_test))/float(batch_size_infer)))
                tq = tqdm(range(length))
                for i in tq:
                    eid = _tdu.get_infer_data(batch_size_infer)
                    if len(eid) == 0:
                        break
                    ner_outputs = model_infer.infer(sess, eid)
                    output = _tdu.get_batch_string_by_index(ner_outputs,current_params.train_type)
                    if (not output.endswith(u'\n')):
                        output += u'\n'
                    final_res += output
                    # batch size 512 * 5 = 2000 + , write into file and clear final string
                    if (i > 0 and i % 5 == 0):
                        infer_f.write((final_res).decode('utf-8'))
                        final_res = ''
                        print('process:%d' % i)
                if (final_res != ''):
                    infer_f.write((final_res).decode('utf-8'))
                infer_f.flush()
                _log_warning('ner infering done.')

            tf.train.write_graph(sess.graph_def, os.path.join(current_params.out_dir), 'ner.pbtxt')
    # check the result
    precision = check_r(label_file=LABEL_FILE_INFER,infer_file=NER_FILE_INFER)
    with _co.open(INFER_PRECISION,'a+',encoding='utf-8') as infer_w:
        try:
            infer_w.write(latest_ckpt + ' : ' + str(precision) + '\n')
        except Exception, e:
            _log_error('compare wrong....')
    return precision

def add_arguments(parser):
    parser.add_argument('--mode', type=str, default='train',
                        help='modes:[train, infer]')
def model_process(exec_mode=_hm.MODE_TRAIN):
    #load dict for data
    _tdu.init_vocab(DATA_VOCAB_FILE, LABEL_VOCAB_FILE)
    gpu_conf = tf.GPUOptions(allow_growth=True)
    if (exec_mode == _hm.MODE_TRAIN):
        train(gpu_conf)
    elif (exec_mode == _hm.MODE_INFER):
        inference(gpu_conf)
    else:
        raise 'wrong choice'

if __name__ == '__main__':
    ner_parser = _ap.ArgumentParser()
    add_arguments(ner_parser)
    FLAGS, _ = ner_parser.parse_known_args()
    _log_info('begin to train...', endline=True)
    model_process(exec_mode=FLAGS.mode)
