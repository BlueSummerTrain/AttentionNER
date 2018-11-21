#!/usr/bin/env Python
# coding=utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import numpy as np
import codecs
import math
import random
import os
import random

import ner.hparams_module as _hm 

main_path = _hm.MAIN_PATH
obj_path = _hm.OBJ_PATH

def shuffle_data(filename,data_list):
	with open(filename,'w') as file_w:
		random.shuffle(data_list)
		file_w.write('\n'.join(data_list).encode('utf-8'))

def create_vocab_data_for_ner(output_data,output_label,train_type,keep_fre=0):
    if output_data == '':
        return
    file_list = []
    if train_type == 'main':
        for item in os.listdir(main_path):
            if os.path.isfile(main_path+item):
                file_list.append(main_path+item)
    if train_type == 'obj':
        for item in os.listdir(obj_path):
            if os.path.isfile(obj_path+item):
                file_list.append(obj_path+item)
    vocab = {}
    vocab_label = {}
    for filename in file_list:
        with codecs.open(filename, 'r', 'utf-8') as file_in:
            data_all = file_in.read().strip().split('\n')
            shuffle_data(filename,data_all)
            for data_it in data_all:
                data_split = data_it.split(',')
                if data_split[0] == '' or data_split[1] == '':
                    continue
                words = data_split[0].strip().split(" ")
                labels = data_split[1].strip().split(" ")
                if len(words)!=len(labels):
                    print data_it
                    continue
                for word in words:
                    if vocab.has_key(word):
                        vocab[word] += 1
                    else:
                        vocab[word] = 1
                for label in labels:
                    if vocab_label.has_key(label):
                        vocab_label[label] += 1
                    else:
                        vocab_label[label] = 1
    sort_list = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
    sort_label_list = sorted(vocab_label.items(), key=lambda x: x[1], reverse=True)

    with codecs.open(output_data, 'w', 'utf-8') as file_w:
        file_w.write('<unk>\n<s>\n</s>\n<padding>\n')
        for i in range(len(sort_list)):
            if (sort_list[i][1] > keep_fre):
                file_w.write(sort_list[i][0].decode('utf-8') + '\n')
    with codecs.open(output_label, 'w', 'utf-8') as file_w:
        file_w.write('<unk>\n<s>\n</s>\n<padding>\n')
        for i in range(len(sort_label_list)):
            if (sort_label_list[i][1] > keep_fre):
                file_w.write(sort_label_list[i][0].decode('utf-8') + '\n')

def sperate_label_data(train_type,fileout):
    if train_type == 'main':
        filename = main_path+'test/'+'Test.txt'
    if train_type == 'obj':
        filename = obj_path+'test/'+'Test.txt'
    with open(filename,'r') as f_r:
        data_list = f_r.read().strip().split('\n')
        data_seq = []
        labels = []
        for data_it in data_list:
            data_it = data_it.split(',')
            if len(data_it[0].split(' ')) != len(data_it[1].split(' ')):
                print data_it[0]
            data_seq.append(data_it[0])
            labels.append(data_it[1])
        fileoutname = fileout+'_'+train_type 
        with open(fileoutname,'w') as f_w:
            f_w.write('\n'.join(labels).encode('utf-8'))

if __name__ == '__main__':
    keep_fre = 0
    if _hm.ner_mode_params.train_type == 'main':
        keep_fre = 1
    create_vocab_data_for_ner('./ner_corpus/'+_hm.WORD_VOC,\
                             './ner_corpus/'+_hm.NER_LABEL,\
                             _hm.ner_mode_params.train_type,\
                             keep_fre=keep_fre)
    sperate_label_data(_hm.ner_mode_params.train_type,'./ner_corpus/test.lf.data')

