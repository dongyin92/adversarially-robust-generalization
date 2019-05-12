from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np

FLAGS = None

def gen_adversary_ex(W, x, y, eps):
  if eps == 0.0:
    return x
  (K, d) = W.shape
  assert(len(x)==d)
  assert(K >= 2)
  assert((y >= 0) and (y < K))
  assert(eps >= 0)
  z = np.dot(W, x)
  W_diff = np.zeros([K, d])
  for i in range(K):
    W_diff[i, :] = W[y, :] - W[i, :]
  score = z
  for i in range(K):
    score[i] += eps * np.linalg.norm(W_diff[i, :], 1)
  s = np.argsort(score)
  ym = s[-1]
  if ym == y:
    ym = s[-2]
  return x - eps * np.sign(W_diff[ym, :])

def main(_):
  
  eps_list = [0.0, 0.006, 0.012, 0.018, 0.024, 0.03]
  batch_size = 100
  num_repetition = 10
  num_step = 10000
  num_train_sample = 1000
  total_train_sample = 55000
  
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
  
  x0 = tf.placeholder(tf.float32, [None, 196])
  W0 = tf.Variable(tf.zeros([196, 10]))
  b0 = tf.Variable(tf.zeros([10]))
  y0 = tf.matmul(x0, W0) + b0
  
  x1 = tf.placeholder(tf.float32, [None, 784])
  W1 = tf.Variable(tf.zeros([784, 10]))
  b1 = tf.Variable(tf.zeros([10]))
  y1 = tf.matmul(x1, W1) + b1
  
  x2 = tf.placeholder(tf.float32, [None, 3136])
  W2 = tf.Variable(tf.zeros([3136, 10]))
  b2 = tf.Variable(tf.zeros([10]))
  y2 = tf.matmul(x2, W2) + b2
  
  # Define loss and optimizer
  y_true = tf.placeholder(tf.float32, [None, 10])
  
  cross_entropy0 = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y0))
  
  cross_entropy1 = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y1))
  
  cross_entropy2 = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y2))
  
  train_step0 = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy0)
  train_step1 = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy1)
  train_step2 = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy2)
  
  correct_prediction0 = tf.equal(tf.argmax(y0, 1), tf.argmax(y_true, 1))
  accuracy0 = tf.reduce_mean(tf.cast(correct_prediction0, tf.float32))
  
  correct_prediction1 = tf.equal(tf.argmax(y1, 1), tf.argmax(y_true, 1))
  accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1, tf.float32))
  
  correct_prediction2 = tf.equal(tf.argmax(y2, 1), tf.argmax(y_true, 1))
  accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2, tf.float32))
  
  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  
  # Train
  # Sample training data.
  train_acc = np.zeros([len(eps_list), 3, num_repetition], dtype=np.float64)
  test_acc = np.zeros([len(eps_list), 3, num_repetition], dtype=np.float64)
  
  # Load data
  print('Loading data...')
  x_train_196 = np.load('mnist_train_images_196.npy')
  x_train_784 = np.load('mnist_train_images_784.npy')
  x_train_3136 = np.load('mnist_train_images_3136.npy')
  y_train_all = np.load('mnist_train_labels.npy')
  
  x_test_196 = np.load('mnist_test_images_196.npy')
  x_test_784 = np.load('mnist_test_images_784.npy')
  x_test_3136 = np.load('mnist_test_images_3136.npy')
  y_test_all = np.load('mnist_test_labels.npy')
  
  
  ######
  d = 196
  for eps_idx in range(len(eps_list)):
    eps = eps_list[eps_idx]
    for r in range(num_repetition):
      print('eps = %.3f, d = %d, r = %d' % (eps, d, r))
      chosen_index = np.random.choice(total_train_sample, num_train_sample,
                                      replace=False)
      x_train = x_train_196[chosen_index, :]
      y_train = y_train_all[chosen_index, :]
      
      for t in range(num_step):
        chosen_indx_iter = np.random.choice(num_train_sample, batch_size,
                                            replace=False)
        batch_xs = x_train[chosen_indx_iter, :]
        batch_ys = y_train[chosen_indx_iter, :]
        W_val = sess.run(W0)
        batch_xs_adv = np.zeros([batch_size, d], dtype=np.float64)
        for s in range(batch_size):
          batch_xs_adv[s, :] = gen_adversary_ex(np.transpose(W_val),
                                            np.transpose(batch_xs[s, :]),
                                            np.argmax(batch_ys[s, :]), eps)
        sess.run(train_step0, feed_dict={x0: batch_xs_adv, y_true: batch_ys})
          
        if t == num_step - 1:
          print('Training finishes!')
          W_val = sess.run(W0)
  
          # Adversarial train error on whole dataset
          print('Check adversarial train accuracy...')

          (num_train_all, image_size) = x_train.shape
          x_train_adv = np.zeros([num_train_all, image_size], dtype=np.float64)
          for i in range(num_train_all):
            x_train_adv[i, :] = gen_adversary_ex(np.transpose(W_val),
                                               np.transpose(x_train[i, :]),
                                               np.argmax(y_train[i, :]),
                                               eps)
          each_train_acc = sess.run(accuracy0, feed_dict={x0: x_train_adv,
                                                          y_true: y_train})
          train_acc[eps_idx, 0, r] = each_train_acc
          print(each_train_acc)
  
          # Adversarial test error on whole dataset
          print('Check adversarial test accuracy...')
          (num_test_all, image_size) = x_test_196.shape
          x_test_adv = np.zeros([num_test_all, image_size], dtype=np.float64)
          for i in range(num_test_all):
            x_test_adv[i, :] = gen_adversary_ex(np.transpose(W_val),
                                                np.transpose(x_test_196[i, :]),
                                                np.argmax(y_test_all[i, :]),
                                                eps)
          each_test_acc = sess.run(accuracy0, feed_dict={x0: x_test_adv,
                                                         y_true: y_test_all})
          test_acc[eps_idx, 0, r] = each_test_acc
          print(each_test_acc)
          np.save('train_acc', train_acc)
          np.save('test_acc', test_acc)
  
  
  ######
  d = 784
  for eps_idx in range(len(eps_list)):
    eps = eps_list[eps_idx]
    for r in range(num_repetition):
      print('eps = %.3f, d = %d, r = %d' % (eps, d, r))
      chosen_index = np.random.choice(total_train_sample, num_train_sample,
                                      replace=False)
      x_train = x_train_784[chosen_index, :]
      y_train = y_train_all[chosen_index, :]
      
      for t in range(num_step):
        chosen_indx_iter = np.random.choice(num_train_sample, batch_size,
                                            replace=False)
        batch_xs = x_train[chosen_indx_iter, :]
        batch_ys = y_train[chosen_indx_iter, :]
        W_val = sess.run(W1)
        batch_xs_adv = np.zeros([batch_size, d], dtype=np.float64)
        for s in range(batch_size):
          batch_xs_adv[s, :] = gen_adversary_ex(np.transpose(W_val),
                                            np.transpose(batch_xs[s, :]),
                                            np.argmax(batch_ys[s, :]), eps)
        sess.run(train_step1, feed_dict={x1: batch_xs_adv, y_true: batch_ys})
          
        if t == num_step - 1:
          print('Training finishes!')
          W_val = sess.run(W1)
  
          # Adversarial train error on whole dataset
          print('Check adversarial train accuracy...')

          (num_train_all, image_size) = x_train.shape
          x_train_adv = np.zeros([num_train_all, image_size], dtype=np.float64)
          for i in range(num_train_all):
            x_train_adv[i, :] = gen_adversary_ex(np.transpose(W_val),
                                               np.transpose(x_train[i, :]),
                                               np.argmax(y_train[i, :]),
                                               eps)
          each_train_acc = sess.run(accuracy1, feed_dict={x1: x_train_adv,
                                                          y_true: y_train})
          train_acc[eps_idx, 1, r] = each_train_acc
          print(each_train_acc)
  
          # Adversarial test error on whole dataset
          print('Check adversarial test accuracy...')
          (num_test_all, image_size) = x_test_784.shape
          x_test_adv = np.zeros([num_test_all, image_size], dtype=np.float64)
          for i in range(num_test_all):
            x_test_adv[i, :] = gen_adversary_ex(np.transpose(W_val),
                                                np.transpose(x_test_784[i, :]),
                                                np.argmax(y_test_all[i, :]),
                                                eps)
          each_test_acc = sess.run(accuracy1, feed_dict={x1: x_test_adv,
                                                         y_true: y_test_all})
          test_acc[eps_idx, 1, r] = each_test_acc
          print(each_test_acc)
          np.save('train_acc', train_acc)
          np.save('test_acc', test_acc)


  ######
  d = 3136
  for eps_idx in range(len(eps_list)):
    eps = eps_list[eps_idx]
    for r in range(num_repetition):
      print('eps = %.3f, d = %d, r = %d' % (eps, d, r))
      chosen_index = np.random.choice(total_train_sample, num_train_sample,
                                      replace=False)
      x_train = x_train_3136[chosen_index, :]
      y_train = y_train_all[chosen_index, :]
      
      for t in range(num_step):
        chosen_indx_iter = np.random.choice(num_train_sample, batch_size,
                                            replace=False)
        batch_xs = x_train[chosen_indx_iter, :]
        batch_ys = y_train[chosen_indx_iter, :]
        W_val = sess.run(W2)
        batch_xs_adv = np.zeros([batch_size, d], dtype=np.float64)
        for s in range(batch_size):
          batch_xs_adv[s, :] = gen_adversary_ex(np.transpose(W_val),
                                            np.transpose(batch_xs[s, :]),
                                            np.argmax(batch_ys[s, :]), eps)
        sess.run(train_step2, feed_dict={x2: batch_xs_adv, y_true: batch_ys})
          
        if t == num_step - 1:
          print('Training finishes!')
          W_val = sess.run(W2)
  
          # Adversarial train error on whole dataset
          print('Check adversarial train accuracy...')

          (num_train_all, image_size) = x_train.shape
          x_train_adv = np.zeros([num_train_all, image_size], dtype=np.float64)
          for i in range(num_train_all):
            x_train_adv[i, :] = gen_adversary_ex(np.transpose(W_val),
                                               np.transpose(x_train[i, :]),
                                               np.argmax(y_train[i, :]),
                                               eps)
          each_train_acc = sess.run(accuracy2, feed_dict={x2: x_train_adv,
                                                          y_true: y_train})
          train_acc[eps_idx, 2, r] = each_train_acc
          print(each_train_acc)
  
          # Adversarial test error on whole dataset
          print('Check adversarial test accuracy...')
          (num_test_all, image_size) = x_test_3136.shape
          x_test_adv = np.zeros([num_test_all, image_size], dtype=np.float64)
          for i in range(num_test_all):
            x_test_adv[i, :] = gen_adversary_ex(np.transpose(W_val),
                                                np.transpose(x_test_3136[i, :]),
                                                np.argmax(y_test_all[i, :]),
                                                eps)
          each_test_acc = sess.run(accuracy2, feed_dict={x2: x_test_adv,
                                                         y_true: y_test_all})
          test_acc[eps_idx, 2, r] = each_test_acc
          print(each_test_acc)
          np.save('train_acc', train_acc)
          np.save('test_acc', test_acc)
  
  np.save('train_acc', train_acc)
  np.save('test_acc', test_acc)
  gen_error = train_acc - test_acc
  np.save('gen_error', gen_error)



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)