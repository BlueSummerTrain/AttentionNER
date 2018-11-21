from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import tensorflow as tf
import hparams_module as _mh
import random

def open_file(filename):
  with open(filename,'r') as data_r:
    return data_r.read().strip().decode('utf-8').split('\n')

def data_inputs(mode):
  data_path = ''
  datas = []
  ner_mode_params = _mh.ner_mode_params()
  
  if ner_mode_params.train_type == 'main':
    if mode == 'train':
      data_path = _mh.MAIN_PATH
    else:
      data_path = _mh.MAIN_PATH + 'test/'
  elif ner_mode_params.train_type == 'obj':
    if mode == 'train':
      data_path = _mh.OBJ_PATH
    else:
      data_path = _mh.OBJ_PATH + 'test/'
  else:
    print 'unknow type data'
    sys.exit(1)
  
  list_files = os.listdir(data_path)
  
  if len(list_files) == 0:
    print 'no train data'
    sys.exit(1)

  for item in list_files:
    if item.endswith('.txt'):
      datas = datas + open_file(data_path+item)
  random.shuffle(datas)

  return datas

def change2ids(src_datas_vocab,tgt_datas_vocab,mode):
  seqs = []
  labels = []
  for item in data_inputs(mode):
    item = item.split(',')
    seq = item[0].split(' ')
    label = item[1].split(' ')
    for j,it in enumerate(seq):
      d_id = 0
      if (src_datas_vocab.has_key(it)):
        d_id = src_datas_vocab[it]
        seq[j] = d_id
    seqs.append(seq)
    if mode == 'train':
      for j,it in enumerate(label):
        d_id = 0
        if (tgt_datas_vocab.has_key(it)):
          d_id = tgt_datas_vocab[it]
          label[j] = d_id
      labels.append(label)
      return np.array(seqs),np.array(labels)
    else:
      return np.array(seqs)

def gen_batch_data_for_train(batch_size):
  datas,labels = change2ids('train')
  data_queue = tf.train.slice_input_producer(datas,labels)
  data_batch, label_batch = \
  tf.train.shuffle_batch(data_queue, batch_size=batch_size, 
    num_threads=1, capacity=2048,min_after_dequeue=1024,allow_smaller_final_batch=True)
  return data_batch,label_batch

def gen_batch_data_for_inference(batch_size):
  datas = change2ids('test')
  data_queue = tf.train.slice_input_producer(datas,labels)
  data_batch, label_batch = \
  tf.train.shuffle_batch(data_queue, batch_size=batch_size, 
    num_threads=1, capacity=2048,min_after_dequeue=1024,allow_smaller_final_batch=True)
  return data_batch,label_batch