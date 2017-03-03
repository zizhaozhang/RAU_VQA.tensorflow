"""
 * @author [Zizhao]
 * @email [zizhao@cise.ufl.edu]
 * @date 2017-02-26 01:23:18
 * @desc [The main model]
"""

from __future__ import absolute_import, division, print_function

import math
import tensorflow as tf

class VQA:
    def __init__(self, config):

        self.config = config

        self.set_default_params(config)
        ## things need to get from input
        self.feed_img_batch = tf.placeholder(
            'float32',
            [None, config.cnnout_h, config.cnnout_w, config.cnnout_dim],
            name='conv_map')
        self.labels = tf.placeholder('int32', [None], name='label')

        self.feed_ques_batch = tf.placeholder(
            'int32', [None, config.max_len], name='ques')
       
        #Sets up the global step Tensor.
        self.global_step = tf.Variable(
            initial_value=0,
            name="global_step",
            trainable=False,
            collections=[
                tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES
            ])

        self.train_mode = True
        self.feed_general_droprate = 0.5
        # the sequence length per sample in the batch
        self.feed_per_seq_length = None  # need to compute before 

        # set some parameters
        self.nHop = config.nHop
        self.batch_size = config.batch_size  

        # intialization of W
        self.initializer = tf.random_uniform_initializer(
            minval=-self.config.initializer_scale,
            maxval=self.config.initializer_scale)

    def training(self):
        self.train_mode = True
        self.feed_general_droprate = 0.5

    def evaluate(self):
        print ('--> model is set to the evaluation mode')
        self.train_mode = False
        self.feed_general_droprate = 0

    def return_feed_placeholder(self):
        summary_merged = tf.summary.merge_all()
        return self.feed_img_batch, self.feed_ques_batch, self.labels, self.global_step, summary_merged

    def set_default_params(self, opt):
        """Set some default parameters than no need to change
        """
        self.config.embed_dim = opt.embed_dim or 200
        self.config.rnn_size = opt.rnn_size or 512
        self.config.nrnn_layer = opt.nrnn_layer or 2
        self.config.rnn_dropout = opt.rnn_dropout or 0.5
        self.config.rnnout_dim = 2 * self.config.rnn_size * self.config.nrnn_layer
        ## MULTIMODAL (ATTENTION)
        self.config.cnnout_dim = opt.cnnout_dim or 512
        self.config.cnnout_w = opt.cnnout_w or 14
        self.config.cnnout_h = opt.cnnout_h or 14
        self.config.cnnout_spat = self.config.cnnout_w * self.config.cnnout_h
        self.config.multfeat_dim = opt.multfeat_dim or 512
        self.config.attfeat_dim = opt.attfeat_dim or 256
        self.config.netout_dim = opt.answer_size
        ## [attlstm] in: {2*multfeat_dim, att_rnn_s_dim} {att_rnn_size, att_rnn_s_dim}
        self.config.att_rnn_size = opt.att_rnn_size or 512
        self.config.att_rnn_nlayer = opt.att_rnn_nlayer or 1
        self.config.att_rnn_dropout = opt.att_rnn_dropout or 0.0
        # TODO: There could be a protential bugs if self.config.att_rnn_nlayer > 1
        assert(self.config.att_rnn_nlayer == 1)
        self.config.att_rnn_s_dim = self.config.att_rnn_size * self.config.att_rnn_nlayer

        # optimization
        self.config.max_grad_norm = opt.max_grad_norm or 0.1
        self.config.initializer_scale = 0.008

    def minimize(self, logits, init_learning_rate, iter_epoch):
        """Minimize funcs
            Input: 
                logits: preds in [[batch_size,self.netout_dim], ...] with nHops returned by self.build_model
                learning_rate: a place_holder
        """

        ## tile the labels for multi hops
        batch_losses = []
        for n in range(self.nHop):
            batch_loss_n = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.labels, logits=logits[n])
            batch_loss_n = tf.reduce_mean(batch_loss_n)
            batch_losses += [batch_loss_n]

            correct_prediction = tf.equal(self.labels, tf.cast(tf.argmax(logits[n], 1), tf.int32))
            accuracy_n = tf.reduce_mean(
                tf.cast(correct_prediction, tf.float32))
            # # write to summary
            tf.summary.scalar('accuracy/acc_' + str(n), accuracy_n)
            tf.summary.scalar('losses/batch_loss_' + str(n), batch_loss_n)

        loss_op = tf.reduce_mean(batch_losses) 
        tf.losses.add_loss(loss_op)
        with tf.name_scope('learning_rate'):
            learning_rate = tf.train.exponential_decay(init_learning_rate, self.global_step, iter_epoch, 0.9, staircase=True)
            tf.summary.scalar('learning_rate', learning_rate)

        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            # TODO: add gaussian noises
            
            # apply gradiet normalization
            tvars = tf.trainable_variables()
            grads, g_norm = tf.clip_by_global_norm(tf.gradients(loss_op, tvars), self.config.max_grad_norm)

            train_op = optimizer.apply_gradients(
                zip(grads, tvars), global_step=self.global_step)
        total_loss = tf.losses.get_total_loss()

        tf.summary.scalar("losses/total_loss", total_loss)
        tf.summary.scalar("parameter_norm", g_norm)
        for var in tf.trainable_variables():
            tf.summary.histogram("parameters/" + var.op.name, var)

        return loss_op, train_op

    def get_per_seq_length(self, sequence):
        """Compute lengh of sequences in self.feed_ques_batch
            Input:
                sequence: [batch, self.max_len], each element is an word id, 
            Output:
                length: length of each batch
        """

        used = tf.sign(tf.abs(sequence))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    def word_embeddings(self, feed_seq_batch):
        """Builds the input sequence embeddings.
        Inputs:
            feed_seq_batch [batch_size, config.max_len]
        Outputs:
            word_embeddings [batch_size, config.embed_dim]
        """
        # compute seq_length
        self.feed_per_seq_length = self.get_per_seq_length(feed_seq_batch)

        # compute embedding
        with tf.variable_scope("word_embedding"):
            seq_embedding = tf.get_variable(
                name="seq_embedding",
                shape=[self.config.vocab_size, self.config.embed_dim],
                initializer=self.initializer)
            word_embeddings = tf.nn.embedding_lookup(seq_embedding,
                                                     feed_seq_batch)
            if self.train_mode:
                word_embeddings = tf.nn.dropout(
                    word_embeddings, keep_prob=self.feed_general_droprate)
            word_embeddings = tf.nn.tanh(word_embeddings)

        return word_embeddings

    def q_embed(self, in_q, in_prev_h):
        """
        Input: 
            in_q [batch_size, config.rnnout_dim]
            in_prev_h [batch_size, config.att_rnn_s_dim]
        Output:
            out_q_embed
        """
        with tf.variable_scope("seq_embedding", reuse=True) as scope:
            if self.train_mode:
                in_q = tf.nn.dropout(in_q, keep_prob=self.feed_general_droprate)

            q_proj = tf.contrib.layers.fully_connected(
                inputs=in_q,
                num_outputs=self.config.multfeat_dim,
                activation_fn=None,
                weights_initializer=self.initializer,
                biases_initializer=None)

            h_proj = tf.contrib.layers.fully_connected(
                inputs=in_prev_h,
                num_outputs=self.config.multfeat_dim,
                activation_fn=None,
                weights_initializer=self.initializer,
                biases_initializer=None)
            out_q_embed = tf.nn.tanh(q_proj + h_proj, name='q_embed')

        return out_q_embed

    def i_embed(self, in_i):
        """Builds the image model subgraph and generates image embeddings.
            Inputs:
                in_i [batch_size, config.cnnout_dim, h, w]: input image conv feature from CNN
            Outputs:
                image_embeddings [batch_size, config.mulfeat_dim, config.cnnout_spat]
        """
        # Map inception output into embedding space.
        with tf.variable_scope("image_embedding", reuse=True) as scope:
            if self.train_mode:
                in_i = tf.nn.dropout(in_i, keep_prob=self.feed_general_droprate)
            i_embed = tf.contrib.layers.conv2d(
                inputs=in_i,
                num_outputs=self.config.multfeat_dim,
                kernel_size=1,
                activation_fn=tf.nn.tanh,
                weights_initializer=self.initializer,
                biases_initializer=None)
            ## reshape to a 2d matrix
            i_embed = tf.reshape(
                i_embed,
                shape=[-1, self.config.cnnout_spat, self.config.multfeat_dim], name='i_embed')

        return i_embed

    def ques_lstm(self, word_embeddings):
        """Build a multi-layer LSTM to read question
            Input:
                word_embeddings [batch_size, config.max_len, config.embed_dim]
            Output:
                ques_encoding [batch_size, config.rnnout_dim]
        """
        nlayer = self.config.nrnn_layer
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(
            num_units=self.config.rnn_size, state_is_tuple=True)
        lstm_cell = tf.contrib.rnn.MultiRNNCell(
            [lstm_cell] * nlayer, state_is_tuple=True)

        with tf.variable_scope(
                'ques_lstm', initializer=self.initializer) as scope:
            # Run the batch of sequence embeddings through the LSTM.
            assert (self.feed_per_seq_length != None)
            _, lstm_state = tf.nn.dynamic_rnn(
                cell=lstm_cell,
                inputs=word_embeddings,
                sequence_length=self.feed_per_seq_length,
                dtype=tf.float32)

            # extract c and h, then concat them all
            ques_encoding_lis = []
            for i in range(nlayer):
                ques_encoding_lis = ques_encoding_lis + [
                    lstm_state[i].c, lstm_state[i].h
                ]
            ques_encoding = tf.concat(ques_encoding_lis, axis=1)

        assert (ques_encoding.shape[1] == self.config.rnnout_dim)

        return ques_encoding

    def build_att_lstm(self, batch_size):
        """Create attention LSTM cell, which will called by self.classifier"""

        self.att_lstm_cell = tf.contrib.rnn.BasicLSTMCell(
            num_units=self.config.att_rnn_size, state_is_tuple=True)
        self.att_lstm_init = self.att_lstm_cell.zero_state(
            batch_size=batch_size, dtype='float32')

    def attlstm(self, time_step, join_input, prev_state):

        with tf.variable_scope('attlstm') as scope:

            hidden, state = self.att_lstm_cell(join_input, prev_state)

        return hidden, state

    def attbycontent(self, in_qfeat, in_ifeat):
        """ Create attention by content
                Input:  
                    in_qfeat [batch_size, config.multfeat_dim]
                    in_ifeat [batch_size, config.multfeat_dim, config.cnnout_spat]
                Output:
                    self.attscore [batch_size, config.cnnout_spat]
            Will call for each time step, so need to reuse variable
        """
        config = self.config
        with tf.variable_scope('attbycontent', reuse=True):
            qfeatatt = tf.contrib.layers.fully_connected(
                in_qfeat,
                num_outputs=config.attfeat_dim,
                weights_initializer=self.initializer,
                activation_fn=None,
                biases_initializer=None)
            qfeatatt = tf.expand_dims(qfeatatt, axis=1)
            
            qfeatatt = tf.tile(qfeatatt, multiples=[1, config.cnnout_spat, 1])
            ifeatproj = tf.expand_dims(in_ifeat, axis=2)
            ifeatproj = tf.contrib.layers.conv2d(
                inputs=ifeatproj,
                num_outputs=config.attfeat_dim,
                kernel_size=1,
                activation_fn=None,
                weights_initializer=self.initializer,
                biases_initializer=None)
            ifeatatt = tf.squeeze(ifeatproj, axis=2)
            addfeat = tf.nn.tanh(ifeatatt + qfeatatt)
            addfeat = tf.expand_dims(addfeat, axis=2)
            attscore = tf.contrib.layers.conv2d(
                inputs=addfeat,
                num_outputs=1,
                kernel_size=1,
                activation_fn=None,
                weights_initializer=self.initializer,
                biases_initializer=None)
            attscore = tf.squeeze(attscore, axis=[2, 3], name='attscore')
        return attscore

    def attbymemory(self, in_attscore, in_prev_h):
        """ Create attention by previous hidden state of attention LSTM
                Input:
                    in_attscore [batch_size, config.cnnout_spat]
                    in_prev_h [batch_size, config.att_rnn_s_dim]
        """
        config = self.config
        with tf.variable_scope('attbymemory', reuse=True):
            attscore_bymem = tf.contrib.layers.fully_connected(
                in_prev_h,
                num_outputs=config.cnnout_spat,
                weights_initializer=self.initializer,
                activation_fn=None,
                biases_initializer=None)
            out_attprob = tf.nn.softmax(attscore_bymem + in_attscore)

        return out_attprob

    def classifier(self, time_step, in_qfeat, in_attfeat, in_attprob,
                   in_prev_h):
        """Classifer for prediction, main component of VQA model
                Input:
                    time_step : a scalar used for attlstm to decide when reuse variables
                    in_qfeat [batch_size, config.multfeat_dim]
                    in_attfeat [batch_size, config.multfeat_dim]
                    in_attprob [batch_size, config.cnnout_spat]
                    in_prev_h [batch_size, config.att_rnn_size]
                Outout:
                    out_score [batch_size, config.netout_dim]
                    tab_lstmout [batch_size, config.att_rnn_s_dim]
        """
        config = self.config
        with tf.variable_scope('classifier') as scope:
            # since we have lstm inside, we can not use reuse=True for variable scope directly
            #TODO: may need to find a better way to allow variable sharing 
            if time_step > 0: 
                tf.get_variable_scope().reuse_variables()
            q_n_att_feat = in_qfeat + in_attfeat
            feat_attprob = tf.contrib.layers.fully_connected(
                in_attprob,
                num_outputs=config.multfeat_dim,
                weights_initializer=self.initializer,
                activation_fn=None,
                biases_initializer=None)
            joint_input = q_n_att_feat + feat_attprob
            # return hidden state only
            tab_lstmout, prev_state = self.attlstm(time_step, joint_input, in_prev_h)
            assert(config.att_rnn_dropout == 0)
            # if self.train_mode:
            #     lstmfeat = tf.nn.dropout(
            #         tab_lstmout, keep_prob=config.att_rnn_dropout)
            lstmfeat = tf.contrib.layers.fully_connected(
                tab_lstmout,
                num_outputs=config.multfeat_dim,
                weights_initializer=self.initializer,
                activation_fn=None,
                biases_initializer=None)
            merge_feat =  lstmfeat + joint_input
            if self.train_mode:
                merge_feat = tf.nn.dropout(
                    merge_feat, keep_prob=self.feed_general_droprate)

            out_score = tf.contrib.layers.fully_connected(
                merge_feat,
                num_outputs=config.netout_dim,
                weights_initializer=self.initializer,
                activation_fn=None,
                biases_initializer=None)
            if self.train_mode == False:
                out_score = tf.nn.softmax(out_score)

        return out_score, prev_state

    def attselect(self, ifeat, att_prob):

        with tf.variable_scope('attselect'):
            context_vector = tf.reduce_sum(ifeat * tf.expand_dims(
                att_prob, axis=2), 1)

        return context_vector

    def multimodel(self, time_step, in_q, in_i, in_prev_state):

        config = self.config
        with tf.variable_scope('multimodel') as scope:
            in_prev_h = in_prev_state.h
            qfeat = self.q_embed(in_q, in_prev_h)
            ifeat = self.i_embed(in_i)
            attscore = self.attbycontent(qfeat, ifeat)
            attprob = self.attbymemory(attscore, in_prev_h)
            attfeat = self.attselect(ifeat, attprob)
            tab_clsout, next_state = self.classifier(time_step, qfeat, attfeat,
                                                 attprob, in_prev_state)
        return tab_clsout, attprob, next_state

    def bulid_model(self):
        """
            Input:

            Output:
                pred [[batch_size, config.netout_dim], ... ]: a list of output for nHop
        """
        config = self.config
        max_len = config.max_len
        # build attention LSTM cell and init_sate
        flow_batch_size = tf.shape(self.feed_ques_batch)[0]
        self.build_att_lstm(flow_batch_size)
        prev_state = self.att_lstm_init
        # compute word embedding
        we = self.word_embeddings(self.feed_ques_batch)
        # compute rnn output of question LSTM
        self.rnn_out = self.ques_lstm(we)
        
        # main loop for all hops
        total_output = []
        with tf.name_scope('hop_iter'):
            for t in range(self.nHop):
                # prev_state is a tuple of c and h
                output, attprob, prev_state = self.multimodel(
                    t, self.rnn_out, self.feed_img_batch, prev_state)
                total_output += [output]

        preds = total_output
        # preds = tf.concat(total_output,axis=1)
        # visualize att_prob
        attention_maps = tf.reshape(
            attprob, shape=[-1, config.cnnout_h, config.cnnout_w, 1])
        attention_maps = tf.image.resize_images(
            attention_maps, size=[224, 224])
        tf.summary.image('attention_maps_' + str(t), attention_maps)

        return preds