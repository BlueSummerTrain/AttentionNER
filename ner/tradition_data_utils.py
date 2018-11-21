# coding=utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import numpy as np
import codecs
import math
import random
import os
import linecache
from rlog import _log_normal, _log_warning, _log_info, _log_error
import hparams_module as _hm

current_params = _hm.ner_mode_params
global src_datas
global tgt_datas
global tgt_intend
global src_datas_vocab
global tgt_datas_vocab
global tgt_datas_rev_vocab
global src_datas_test
global tgt_datas_test

src_datas_vocab = {}
tgt_datas_vocab = {}
tgt_datas_rev_vocab = {}

SRC_MAX_LENGTH = 40
TGT_MAX_LENGTH = SRC_MAX_LENGTH
NMT_UNK_ID = 0
NMT_SOS_ID = 1
NMT_EOS_ID = 2
PADDING = 3

BUCKET_ID = 0
BATCH_STEP = 0
BATCH_STEP_test = 0

def purify_datas(lines):
        # sort each line into list
    length = len(lines)
    if (length < 1):
        return
    for i in range(length):
        line = lines[i]
        line = line.strip()
        if (line.endswith(u'\n')):
            line = line[:-1]
        line = line.split(u' ')
        lines[i] = line
    return lines

def init_dataset(data_seq, labels, mode='train'):
    global src_datas
    global tgt_datas
    global src_datas_test
    global tgt_datas_test
    if (mode == 'train'):
        src_datas = data_seq
        tgt_datas = labels
        return len(src_datas)
    else:
        src_datas_test = purify_datas(data_seq)
        tgt_datas_test = purify_datas(labels)



def init_vocab(src_vocab_file, tgt_vocab_file):
        # init the vocab from vocab_corpus
    global src_datas_vocab
    global tgt_datas_vocab
    global tgt_datas_rev_vocab

    if (src_vocab_file != ''):
        with codecs.open(src_vocab_file, 'r', encoding='utf-8') as src_f:
            src_vocab_lines = src_f.readlines()
            src_temp_vocab = {}
            for line in src_vocab_lines:
                line = line.strip()
                if (line.endswith(u'\n')):
                    line = line[:-1]
                src_temp_vocab[line] = len(src_temp_vocab)
            src_datas_vocab = src_temp_vocab
            del src_temp_vocab

    if(tgt_vocab_file != ''):
        with codecs.open(tgt_vocab_file, 'r', encoding='utf-8') as tgt_f:
            tgt_vocab_lines = tgt_f.readlines()
            tgt_temp_vocab = {}
            for line in tgt_vocab_lines:
                line = line.strip()
                if (line.endswith(u'\n')):
                    line = line[:-1]
                tgt_temp_vocab[line] = len(tgt_temp_vocab)
            tgt_datas_vocab = tgt_temp_vocab
            del tgt_temp_vocab

            temp_rev_vocab = {}
            for (i, j) in zip(tgt_datas_vocab.keys(), tgt_datas_vocab.values()):
                temp_rev_vocab[j] = i
            tgt_datas_rev_vocab = temp_rev_vocab
            del temp_rev_vocab


def check_if_hits_label(e):
    global tgt_datas_vocab
    # print tgt_datas_vocab
    if current_params.train_type == 'main':
        if (e != tgt_datas_vocab['I_NAME'] and \
            e != tgt_datas_vocab['B_NAME'] and \
            e != tgt_datas_vocab['E_NAME'] and \
            e != tgt_datas_vocab['I_ATV'] and \
            e != tgt_datas_vocab['E_NUM'] and \
            e != tgt_datas_vocab['B_NUM'] and \
            e != tgt_datas_vocab['B_ATV'] and \
            e != tgt_datas_vocab['E_ATV'] and \
            e != tgt_datas_vocab['I_NUM'] and \
            e != tgt_datas_vocab['S_NUM'] and \
            e != tgt_datas_vocab['S_NAME']):
            return False
        else:
            return True
    else:
        if (e != tgt_datas_vocab['S_OBJ']):
            return False
        else:
            return True
def check_obj_labels(noise):
    global src_datas_vocab
    NAME = u'\u1401'
    NUM = u'\u1405'
    TI = u'\u1402'
    TV_CH = u'\u1573'
    noise_data = list(src_datas_vocab.keys())[list(src_datas_vocab.values()).index(noise)]
    if noise == NAME or noise == NUM or noise == TI or noise == TV_CH:
        return True
    else:
        return False

def add_noise(src, tgt):
    global NOISE_AUGMENT
    global src_datas_vocab
    global tgt_datas_vocab
    global tgt_datas_rev_vocab

    # _tgt = []
    # for _t in tgt:
    #     _tgt.append(tgt_datas_rev_vocab[_t])
    # print 'r_tgt1', _tgt

    _odds = 0.6180339
    if (current_params.add_noise and np.random.randint(1, 101) < (_odds * 100)):
        _unk_odds = 1 - 0.6180339
        # 1 ~ 3
        noise_count = np.random.randint(1, 4)
        noise_count_add = 0
        if (np.random.randint(1, 101) < _unk_odds * 100 ):
            while (noise_count_add < noise_count):
                _pos = np.random.randint(0, len(src)+1)
                if (_pos == len(src)):
                    if (np.random.randint(0, 2) == 0):
                        src.insert(0, NMT_UNK_ID)
                        tgt.insert(0, tgt_datas_vocab['O'])
                    else:
                        src.insert(len(src), NMT_UNK_ID)
                        tgt.insert(len(src), tgt_datas_vocab['O'])
                    noise_count_add += 1
                elif (check_if_hits_label(tgt[_pos]) == False):
                    src.insert(_pos, NMT_UNK_ID)
                    tgt.insert(_pos, tgt_datas_vocab['O'])
                    noise_count_add += 1
        else:
            while (noise_count_add < noise_count):
                _pos = np.random.randint(0, len(src)+1)
                # print '_pos', _pos
                if (_pos == len(src)):
                    noise = np.random.randint(0, len(src_datas_vocab))
                    if (noise == NMT_EOS_ID or noise == NMT_SOS_ID or noise == PADDING or check_obj_labels(noise)):
                        noise = NMT_UNK_ID
                    if (np.random.randint(0, 2) == 0):
                        src.insert(0, noise)
                        tgt.insert(0, tgt_datas_vocab['O'])
                    else:
                        src.insert(len(src), noise)
                        tgt.insert(len(src), tgt_datas_vocab['O'])
                    noise_count_add += 1
                elif (check_if_hits_label(tgt[_pos]) == False):
                    noise = np.random.randint(0, len(src_datas_vocab))
                    if (noise == NMT_EOS_ID or noise == NMT_SOS_ID or noise == PADDING or check_obj_labels(noise)):
                        noise = NMT_UNK_ID
                    src.insert(_pos, noise)
                    tgt.insert(_pos, tgt_datas_vocab['O'])
                    noise_count_add += 1
        # print 'add noise'
        # print 'src', src
        # print 'tgt', tgt
        # _tgt = []
        # for _t in tgt:
        #     _tgt.append(tgt_datas_rev_vocab[_t])
        # print 'r_tgt2', _tgt
    return src, tgt

def get_batch_data_for_train(batch_size=16):
    global src_datas
    global tgt_datas
    global src_datas_vocab
    global tgt_datas_vocab
    global BATCH_STEP

    assert len(src_datas) == len(tgt_datas), 'src data not match with tgt data'

    bucket_length = len(src_datas)
    # _log_info('get data size:%d' %(bucket_length))
    inner_src_datas = src_datas
    inner_tgt_datas = tgt_datas
    encoder_input_data = []
    decoder_output_data = []
    max_seq_length = 0
    for i in range(BATCH_STEP, min(bucket_length, BATCH_STEP+batch_size)):

        #get input data to dict id
        sub_src_input_data = inner_src_datas[i].split(' ')
        sub_src_input_data = sub_src_input_data[:SRC_MAX_LENGTH - 1]
        for j,item in enumerate(sub_src_input_data):
            d_id = 0
            if (src_datas_vocab.has_key(item)):
                d_id = src_datas_vocab[item]
            sub_src_input_data[j] = d_id


        #get input label to dict id
        sub_tgt_output_data = inner_tgt_datas[i].split(' ')
        sub_tgt_output_data = sub_tgt_output_data[:TGT_MAX_LENGTH - 1]
        for j,item in enumerate(sub_tgt_output_data):
            d_id = 0
            if (tgt_datas_vocab.has_key(item)):
                d_id = tgt_datas_vocab[item]
            sub_tgt_output_data[j] = d_id

        # add noise
        sub_src_input_data, sub_tgt_output_data = add_noise(sub_src_input_data, sub_tgt_output_data)
        sub_src_input_data = sub_src_input_data + [NMT_EOS_ID]
        encoder_input_data.append(sub_src_input_data)
        if len(sub_src_input_data) > max_seq_length:
            max_seq_length = len(sub_src_input_data)
        sub_tgt_output_data = sub_tgt_output_data + [NMT_EOS_ID] 
        decoder_output_data.append(sub_tgt_output_data)

    BATCH_STEP += batch_size
    batch_max_seq_length = max_seq_length # ner input same as output

    if (BATCH_STEP > bucket_length):
        padding_data_len = batch_size - ( BATCH_STEP - bucket_length )
        BATCH_STEP = 0
    else:
        padding_data_len = batch_size
    
    if len(encoder_input_data) ==0:
        return []
    #pading data to max data length
    for i in range(padding_data_len):
        encode_input = encoder_input_data[i]
        decode_output = decoder_output_data[i]
        if len(encode_input) < batch_max_seq_length:
            encoder_input_data[i] = encode_input + [PADDING]*(batch_max_seq_length - len(encode_input))
        if len(decode_output) < batch_max_seq_length:
            decoder_output_data[i] = decode_output +[PADDING]*(batch_max_seq_length - len(decode_output))

    encoder_input_data = np.array(encoder_input_data, dtype=np.int32)
    decoder_output_data = np.array(decoder_output_data, dtype=np.int32)

    return [encoder_input_data, decoder_output_data]

def get_infer_data(batch_size=2):
    global src_datas_test
    global src_datas_vocab
    global BATCH_STEP_test

    encoder_input_data = []
    max_seq_length = 0
    data_length = len(src_datas_test)
    for i in range(BATCH_STEP_test, min(data_length, BATCH_STEP_test+batch_size)):
        data = src_datas_test[i]
        data = data[:SRC_MAX_LENGTH - 1]
        for j in range(len(data)):
            d = data[j]
            d_id = 0
            if (src_datas_vocab.has_key(d)):
                d_id = src_datas_vocab[d]
            data[j] = d_id
        data += [NMT_EOS_ID]
        if len(data) > max_seq_length:
            max_seq_length = len(data)
        encoder_input_data.append(data)
    
    BATCH_STEP_test += batch_size
    if BATCH_STEP_test > data_length:
        padding_data_len = batch_size - ( BATCH_STEP_test - data_length )
        BATCH_STEP_test = 0
    else:
        padding_data_len = batch_size

    if len(encoder_input_data) ==0:
        return []
    for i in range(padding_data_len):
        seid = encoder_input_data[i]
        seid_spl = max_seq_length - len(seid)
        if (seid_spl > 0):
            seid += [PADDING] * (seid_spl)
        encoder_input_data[i] = seid
    
    encoder_input_data = np.array(encoder_input_data, dtype=np.int32)
    return encoder_input_data

def get_batch_string_by_index(ner_output,train_type):
    global tgt_datas_rev_vocab
    if (tgt_datas_rev_vocab == None):
        _log_error('tgt_datas_rev_vocab not initialized.')
        return ''
    rnn_output = ner_output
    output_len = len(rnn_output)
    data_out = []
    for i in range(output_len):
        out = rnn_output[i]
        res_str_list = []
        for j in range(len(out)):
            t = tgt_datas_rev_vocab[out[j]]
            if (t != u'</s>' and t != u'<padding>'):
                if t == u'<unk>':
                    t = ''
                res_str_list.append(t)
            else:
                break
        data_out.append(' '.join(res_str_list))
    return '\n'.join(data_out)

def convert_to_ids_by_single(sentence):
    global src_datas_vocab
    if (sentence == None or sentence == ''):
        _log_error('invald sentence.')
        return None, None
    seq = []
    sentence = sentence[:SRC_MAX_LENGTH - 1]
    for word in sentence:
        d_id = 0
        if (src_datas_vocab.has_key(word)):
            d_id = src_datas_vocab[word]
        seq.append(d_id)
    seq += [NMT_EOS_ID]
    if len(seq) < SRC_MAX_LENGTH:
        seq = seq + (SRC_MAX_LENGTH - len(seq))*[PADDING]
    seq = np.array(seq, dtype=np.int32)
    return seq

def return_back_seq(eid):
    seq_data=[]
    for it in eid:
        s_data = list(src_datas_vocab.keys())[list(src_datas_vocab.values()).index(it)]
        seq_data.append(s_data)
    return ''.join(seq_data)