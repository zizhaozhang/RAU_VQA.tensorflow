"""
 * @author [Zizhao]
 * @email [zizhao@cise.ufl.edu]
 * @date 2017-03-02 02:06:43
 * @desc [A tensorflow version of RAU_VQA (https://github.com/HyeonwooNoh/RAU_VQA)]
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import argparse
import os
from model.vqa import VQA
from data_loader import VQADataLoader
from opts import OPT


FLAGS = tf.app.flags.FLAGS
# general parameters
tf.flags.DEFINE_string("batch_size", 32,
                       "batch size")             
tf.flags.DEFINE_string("log_dir", "./checkpoint/",
                       "log_dir")
tf.flags.DEFINE_float("init_learning_rate", 0.001,
                       "learning rate")
tf.flags.DEFINE_integer("epoch", 50, "Number of training epochs.")
tf.flags.DEFINE_integer("log_every_epoch", 1,
                        "Frequency at which loss and global step are logged.")
tf.flags.DEFINE_string("checkpoint_path", "./checkpoint/",
                       "if load checkpoint")  
## CNN param
tf.flags.DEFINE_integer("cnnout_dim", 512, "Number of feature map of CNN outputs.")                    
tf.logging.set_verbosity(tf.logging.INFO)


def compute_accuracy(preds):
    #TODO: compute open-ended and multi-choice accuracy
    pass

## configure data loader
data_loader = VQADataLoader(FLAGS)
# configure opts
opts = OPT(FLAGS, data_loader)
## configure models
model = VQA(opts)


def main(_):

    # configuration session
    config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # define saver
    saver = tf.train.Saver(max_to_keep=20)

    preds = model.bulid_model()
    loss_op, train_op = model.minimize(preds, FLAGS.init_learning_rate, data_loader.get_iter_epoch())
    imgbatch, quesbatch, labels, global_step, summary_merged = model.return_feed_placeholder()
    # print ('~~~~~~~~~~~~~~~~~', summary_merged)
    iter_epoch = data_loader.get_iter_epoch()
    tot_iter = iter_epoch * FLAGS.epoch
    
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    with tf.Graph().as_default(), sess.as_default():
       
        # restore from a checkpoint if exists
        # the name_scope can not change 
        if len(FLAGS.load_from_checkpoint) > 0:
            saver.restore(sess, os.path.join('checkpoints', FLAGS.load_from_checkpoint))
            print ('--> load from checkpoint '+ FLAGS.load_from_checkpoint)

        # system setup
        if os.path.isdir(FLAGS.log_dir): os.system('rm ' + FLAGS.log_dir + '/*')
        train_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
        
        start = global_step.eval()
        for it in range(start, tot_iter):
            batch = data_loader.load_batch()

            # tensorflow wants a different tensor order
            feed_dict = {   imgbatch: batch['feat'].transpose((0,2,3,1)),
                            quesbatch: batch['question'],
                            labels: batch['answer']
                        }

            summary, _, loss = sess.run([summary_merged, train_op, loss_op], feed_dict=feed_dict)
                                        
            global_step.assign(it).eval()
            train_writer.add_summary(summary, it)
            if it % 10 == 0 : 
                print ('epoch %f: loss=%f' % (float(it)/iter_epoch, loss))
            if it % iter_epoch == 0:
                # do evaluation
                model.evaluate()
                batch = data_loader.load_batch_test()

                # tensorflow wants a different tensor order
                feed_dict = {   imgbatch: batch['feat'].transpose((0,2,3,1)),
                                quesbatch: batch['question'],
                                labels: batch['answer']
                            }

                summary, oe_preds = sess.run([summary_merged, preds], feed_dict=feed_dict)
                compute_accuracy(oe_preds)

                saver.save(sess, FLAGS.checkpoint_path, global_step=global_step)
                print ('save a checkpoint at '+ FLAGS.checkpoint_path+'-'+str(it))

if __name__ == "__main__":
    tf.app.run()