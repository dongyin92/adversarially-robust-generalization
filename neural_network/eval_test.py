from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import json
import math
import os
import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from model import Model
from pgd_attack import LinfPGDAttack

# Global constants
with open('config.json') as config_file:
  config = json.load(config_file)
num_eval_examples = config['num_eval_examples']
eval_batch_size = config['eval_batch_size']
eval_on_cpu = config['eval_on_cpu']

model_dir = config['model_dir']

# Set upd the data, hyperparameters, and the model
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

if eval_on_cpu:
  with tf.device("/cpu:0"):
    model = Model()
    attack = LinfPGDAttack(model, 
                           config['epsilon'],
                           config['k'],
                           config['a'],
                           config['random_start'],
                           config['loss_func'])
else:
  model = Model()
  attack = LinfPGDAttack(model, 
                         config['epsilon'],
                         config['k'],
                         config['a'],
                         config['random_start'],
                         config['loss_func'])

global_step = tf.contrib.framework.get_or_create_global_step()

# Setting up the Tensorboard and checkpoint outputs
if not os.path.exists(model_dir):
  os.makedirs(model_dir)
eval_dir = os.path.join(model_dir, 'eval')
if not os.path.exists(eval_dir):
  os.makedirs(eval_dir)

last_checkpoint_filename = ''
already_seen_state = False

saver = tf.train.Saver()
summary_writer = tf.summary.FileWriter(eval_dir)

# A function for evaluating a single checkpoint
def evaluate_checkpoint(filename):
  with tf.Session() as sess:
    # Restore the checkpoint
    saver.restore(sess, filename)

    # Iterate over the samples batch-by-batch
    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
    total_xent_nat = 0.
    total_xent_adv = 0.
    total_corr_nat = 0
    total_corr_adv = 0

    for ibatch in range(num_batches):
      bstart = ibatch * eval_batch_size
      bend = min(bstart + eval_batch_size, num_eval_examples)

      x_batch = mnist.test.images[bstart:bend, :]
      y_batch = mnist.test.labels[bstart:bend]

      dict_nat = {model.x_input: x_batch,
                  model.y_input: y_batch}

      x_batch_adv = attack.perturb(x_batch, y_batch, sess)

      dict_adv = {model.x_input: x_batch_adv,
                  model.y_input: y_batch}

      cur_corr_nat, cur_xent_nat = sess.run(
                                      [model.num_correct,model.xent],
                                      feed_dict = dict_nat)
      cur_corr_adv, cur_xent_adv = sess.run(
                                      [model.num_correct,model.xent],
                                      feed_dict = dict_adv)

      total_xent_nat += cur_xent_nat
      total_xent_adv += cur_xent_adv
      total_corr_nat += cur_corr_nat
      total_corr_adv += cur_corr_adv

    avg_xent_nat = total_xent_nat / num_eval_examples
    avg_xent_adv = total_xent_adv / num_eval_examples
    acc_nat = total_corr_nat / num_eval_examples
    acc_adv = total_corr_adv / num_eval_examples

    summary = tf.Summary(value=[
          tf.Summary.Value(tag='xent adv eval', simple_value= avg_xent_adv),
          tf.Summary.Value(tag='xent adv', simple_value= avg_xent_adv),
          tf.Summary.Value(tag='xent nat', simple_value= avg_xent_nat),
          tf.Summary.Value(tag='accuracy adv eval', simple_value= acc_adv),
          tf.Summary.Value(tag='accuracy adv', simple_value= acc_adv),
          tf.Summary.Value(tag='accuracy nat', simple_value= acc_nat)])
    summary_writer.add_summary(summary, global_step.eval(sess))

    print('natural: {:.2f}%'.format(100 * acc_nat))
    print('adversarial: {:.2f}%'.format(100 * acc_adv))
    print('avg nat loss: {:.4f}'.format(avg_xent_nat))
    print('avg adv loss: {:.4f}'.format(avg_xent_adv))
    return acc_nat, acc_adv


test_acc_nat_list = []
test_acc_adv_list = []

#cur_checkpoint = tf.train.latest_checkpoint(model_dir)
checkpoint_num_list = ['99000']
checkpoint_file_list = []
for each_num in checkpoint_num_list:
  checkpoint_file_list.append(model_dir + '/checkpoint-' + each_num)

for cur_checkpoint in checkpoint_file_list:
  print('\nCheckpoint {}, evaluating ...   ({})'.format(cur_checkpoint,
                                                        datetime.now()))
  print('......')
  print(cur_checkpoint)
  print('......')
  acc_nat, acc_adv = evaluate_checkpoint(cur_checkpoint)
  test_acc_nat_list.append(acc_nat)
  test_acc_adv_list.append(acc_adv)
  np.save('test_acc_nat_list', test_acc_nat_list)
  np.save('test_acc_adv_list', test_acc_adv_list)
  print(test_acc_nat_list)
  print(test_acc_adv_list)
