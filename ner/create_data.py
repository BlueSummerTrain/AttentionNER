#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import numpy as np
import sys
import os
import random
import hparams_module as _hm
import rlog

current_params = _hm.ner_mode_params
main_path = '../'+_hm.MAIN_PATH
obj_path = '../'+_hm.OBJ_PATH

def open_file(filename,start):
	small_file = False
	read_len = current_params.read_data_len
	with open(filename,'rb') as file_r:
		data_list = file_r.read().decode('utf-8').strip().split('\n')
		sample_data_len = len(data_list)

		if start < 0:#for test or obj corpus
			return data_list,small_file

		if sample_data_len < read_len:
			small_file = True
			if start == 0:
				random.shuffle(data_list)
				rlog._log_warning("small file %s,read all"%(filename.split('/')[-1]))
				with open(filename,'w') as file_w:
					file_w.write('\n'.join(data_list).encode('utf-8'))
			return data_list,small_file

		if start == 0:
			rlog._log_warning("starting shuffle %s"%(filename.split('/')[-1]))
			random.shuffle(data_list)
			with open(filename,'w') as file_w:
				file_w.write('\n'.join(data_list).encode('utf-8'))
		if (start+1)*read_len <= sample_data_len:
			ner_data = data_list[start*read_len:(start+1)*read_len]
		else:
			if start*read_len < sample_data_len:
				ner_data = data_list[start*read_len:sample_data_len]
			else:
				if read_len/2 < sample_data_len:
					ner_data = random.sample(data_list,read_len/2)
				else:
					ner_data = data_list
	return ner_data,small_file

def get_all_train_data(train_type):

	train_list = []
	if train_type == 'main':
		for item in os.listdir(main_path):
			if os.path.isfile(main_path+item):
				data_list,_ = open_file(main_path+item,-1)
				train_list = train_list + data_list
	if train_type == 'obj':
		for item in os.listdir(obj_path):
			if os.path.isfile(obj_path+item):
				data_list,_ = open_file(main_path+item,-1)
				train_list = train_list + data_list

	random.shuffle(train_list)

	labels = []
	data_seq = []
	for item in train_list:
		item = item.strip().split(',')
		if len(item)<=1:
			continue
		words_test = item[0].strip().split(' ')
		labels_test = item[1].strip().split(' ')
		if item[0] == '' or item[1] == '' or len(words_test)!=len(labels_test):
			continue
		data_seq.append(item[0])
		labels.append(item[1])
	return data_seq,labels,_,_,_,_

def train_data(train_type,start):
	train_list = []
	small_list = []
	count_num = 0
	if train_type == 'main':
		for item in os.listdir(main_path):
			if os.path.isfile(main_path+item):
				data_list,small_file = open_file(main_path+item,start)
				if small_file:
					small_list = small_list + data_list
				else:
					count_num = count_num + 1
					train_list = train_list + data_list
	if train_type == 'obj':
		for item in os.listdir(obj_path):
			if os.path.isfile(obj_path+item):
				data_list,small_file = open_file(obj_path+item,-1)
				train_list = train_list + data_list

	random.shuffle(train_list)
	random.shuffle(small_list)

	labels = []
	data_seq = []
	for item in train_list:
		item = item.strip().split(',')
		if len(item)<=1:
			continue
		words_test = item[0].strip().split(' ')
		labels_test = item[1].strip().split(' ')
		if item[0] == '' or item[1] == '' or len(words_test)!=len(labels_test):
			continue
		data_seq.append(item[0])
		labels.append(item[1])

	labels_small = []
	data_seq_small = []
	if len(small_list) > 0:
		for item in small_list:
			item = item.strip().split(',')
			if len(item)<=1:
				continue
			words_test = item[0].strip().split(' ')
			labels_test = item[1].strip().split(' ')
			if item[0] == '' or item[1] == '' or len(words_test)!=len(labels_test):
				continue
			data_seq_small.append(item[0])
			labels_small.append(item[1])

	return data_seq,labels,data_seq_small,labels_small,len(train_list),count_num

def test_data(train_type):
	train_dict = []
	main_path_test = main_path+'test/'
	obj_path_test = obj_path+'test/'
	if train_type == 'main':
		for item in os.listdir(main_path_test):
			if os.path.isfile(main_path_test+item):
				data_list,_ = open_file(main_path_test + item,-1)
				train_dict = train_dict + data_list
	if train_type == 'obj':
		for item in os.listdir(obj_path_test):
			if os.path.isfile(obj_path_test + item):
				obj_data_list,_ = open_file(obj_path_test + item,-1)
				train_dict = train_dict + obj_data_list
	labels=[]
	data_seq=[]
	for item in train_dict:
		item = item.strip().split(',')
		data_seq.append(item[0])
		labels.append(item[1])
	return data_seq,labels
		
if __name__ == '__main__':
	pass
