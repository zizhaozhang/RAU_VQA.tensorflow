"""
 * @author [Zizhao]
 * @email [zizhao@cise.ufl.edu]
 * @date 2017-02-28 10:56:38
 * @desc [description]
"""

class OPT():
    def __init__(self, input_opt, data_op):
        # define all parameters
        self.embed_dim = None
        self.rnn_size = None
        self.nrnn_layer = None
        self.rnn_dropout = None
        self.cnnout_dim = input_opt.cnnout_dim
        self.cnnout_w = 14
        self.cnnout_h = 14
        self.multfeat_dim = None
        self.attfeat_dim = None
        self.answer_size = 1000
        self.att_rnn_size = None
        self.att_rnn_nlayer = None
        self.att_rnn_dropout = None

        self.max_grad_norm = None
        self.batch_size = input_opt.batch_size
        self.initializer_scale = 0.08


        ## for test, feak params
        self.answer_size = data_op.answer_size
        self.max_len = data_op.max_sentence_len
        self.nHop = 8
        self.vocab_size = data_op.vocab_size

        self.shuffle = True
