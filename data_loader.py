"""
 * @author [Zizhao]
 * @email [zizhao@cise.ufl.edu]
 * @date 2017-03-02 09:24:22
 * @desc [description]
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import numpy as np
import h5py 
import torchfile, random

class VQADataLoader():
    def __init__(self, opt):

        root = './data/'
        self.vqa_dir = os.path.join(root,'VQA_prepro/data_train-val_test-dev')

        self.img_feat_path = os.path.join(root,'vqa_VGG16Conv_pool5_448/feat_448x448/')

        self.batch_size = opt.batch_size
        self.ifshuffle = True
        self.feat_dim = opt.cnnout_dim

        self.feat_h, self.feat_w = 14, 14

        self.train_mode = True

        # load hdf5 file
        self.load_train_ques()

        # compute vocab size
        info_json = os.path.join(self.vqa_dir, 'data_prepro.json')
        with open(info_json,'r') as f:
            json_file = json.load(f)
        
        # build dataset info
        self.vocab_size = len(json_file['ix_to_word'])
        self.answer_size = len(json_file['ix_to_ans'])
        ## vocab dict and vocab map construction
        self.vocab_dict = {0:'ZEROPAD'}
        self.answer_dict = dict()
        self.vocab_map = {'ZEROPAD': 0}
        self.answer_map = dict()
        for i, w in json_file['ix_to_word'].iteritems():
            self.vocab_dict[int(i)] = w
            self.vocab_map[w] = int(i) 
        for i , w in json_file['ix_to_ans'].iteritems():
            self.answer_dict[int(i)] = w
            self.answer_map[w] = int(i)
        
        self.img_train = json_file['unique_img_train']
        self.img_test = json_file['unique_img_test']
        self.seq_len = self.max_sentence_len = self.train_ques['question'].shape[1]
        self.ex_num_train = self.train_ques['question'].shape[0]

        # # file list
        self.img_list_train = dict()
        for i, idx in enumerate(self.train_ques['img_list']):
            self.img_list_train[i] = self.img_train[idx-1]
        self.img_list_test = dict()
        for i, idx in enumerate(self.test_ques['img_list']):
            self.img_list_test[i] = self.img_test[idx-1]

        # setup reader info
        self.batch_order = range(self.ex_num_train)
        self.iter_per_epoch = int(len(self.batch_order) / self.batch_size)
        self.shuffle()

    
    def load_batch(self):

        if self.train_mode:
            x_ques = self.train_ques
        else:
            x_ques = self.test_ques

        sIdx = self.batch_index 
        eIdx = self.batch_index + self.batch_size
        bIndx = [self.batch_order[s] for s in range(sIdx, eIdx)]
        batch_q_len = x_ques['lengths_q'].copy()
        batch_qid = x_ques['question_id'][bIndx].copy()
        batch_q = x_ques['question'][bIndx].copy()

        # feature matrx
        batch_feat = np.zeros(shape=(self.batch_size, self.feat_dim, self.feat_w, self.feat_h))

        if self.train_mode:
            batch_a = x_ques['answers'][bIndx].copy()
        else:
            batch_a = x_ques['mc_ans'][bIndx].copy()

        ## load feature batch
        for i, idx in enumerate(bIndx):
            img_name = self.img_list_train[idx]
            _, img_name = os.path.split(img_name)
            img_name, img_ext = os.path.splitext(img_name)
            feat_path = os.path.join(self.img_feat_path, img_name+'.t7')
            feature = torchfile.load(feat_path)
            batch_feat[i] = feature

            self.batch_index += self.batch_size
            if (self.batch_index + self.batch_size) > self.ex_num_train:
                self.shuffle()
        
        train_batch = {
            'feat': batch_feat,
            'question': batch_q,
            'ques_len': batch_q_len,
            'answer': batch_a,
            'ques_id': batch_qid
        }
        return train_batch

    def get_iter_epoch(self):
        return self.iter_per_epoch

    def shuffle(self):
        if self.ifshuffle:
            self.batch_index = 0
            random.shuffle(self.batch_order)
            assert(self.batch_order[0] != 0)


    def load_train_ques(self):
        
        h5_file_path = os.path.join(self.vqa_dir, 'data_prepro.h5')
        h5_file_data = h5py.File(h5_file_path)

       
        self.train_ques = {
            'question': np.asarray(h5_file_data['ques_train']), # make zero padding to one
            'lengths_q': np.asarray(h5_file_data['ques_length_train']),
            'img_list': np.asarray(h5_file_data['img_pos_train']),
            'question_id':  np.asarray(h5_file_data['question_id_train']),
            'answers': np.asarray(h5_file_data['answers'])
        }
        
        all_items = dict([(key,1) for key in h5_file_data.keys()])

        if all_items.has_key('datatype_train'):
            self.train_ques['datatype'] = np.asarray(h5_file_data['datatype_train'])
        else:
            self.train_ques['datatype'] = self.train_ques['answers'].copy()
            self.train_ques['datatype'][:] = 1

        self.test_ques = {
            'question': np.asarray(h5_file_data['ques_test']), # make zero padding to one
            'lengths_q': np.asarray(h5_file_data['ques_length_test']),
            'img_list': np.asarray(h5_file_data['img_pos_test']),
            'question_id':  np.asarray(h5_file_data['question_id_test']),
            'mc_ans': np.asarray(h5_file_data['MC_ans_test'])
        }
        self.test_ques['dataype'] = self.test_ques['question_id'].copy()
        self.test_ques['dataype'][:] = 1

