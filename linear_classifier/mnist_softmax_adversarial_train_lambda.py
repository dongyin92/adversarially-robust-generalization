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
  lambda_list = [0.0, 0.001, 0.002]
  batch_size = 100
  num_repetition = 10
  num_step = 10000
  num_train_sample = 1000
  total_train_sample = 55000
  
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
  x = tf.placeholder(tf.float32, [None, 784])
  W = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))
  lmda = tf.placeholder(tf.float32, shape=())
  y = tf.matmul(x, W) + b

  # Define loss and optimizer
  y_true = tf.placeholder(tf.float32, [None, 10])  

  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y))
  
  loss = cross_entropy + lmda * tf.norm(W, ord=1)
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
  
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_true, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()

  # Train
  # Sample training data.
  train_acc = np.zeros([len(eps_list), len(lambda_list), num_repetition],
                       dtype=np.float64)
  test_acc = np.zeros([len(eps_list), len(lambda_list), num_repetition],
                      dtype=np.float64)
  for eps_idx in range(len(eps_list)):
    eps = eps_list[eps_idx]
    for lambda_idx in range(len(lambda_list)):
      lmda_val = lambda_list[lambda_idx]
      for r in range(num_repetition):
        print('eps = %.3f, lambda = %.3f, r = %d' % (eps, lmda_val, r))
        chosen_index = np.random.choice(total_train_sample, num_train_sample,
                                        replace=False)
        x_train = mnist.train.images[chosen_index, :]
        y_train = mnist.train.labels[chosen_index, :]

        for t in range(num_step):
          chosen_indx_iter = np.random.choice(num_train_sample, batch_size,
                                              replace=False)
          batch_xs = x_train[chosen_indx_iter, :]
          batch_ys = y_train[chosen_indx_iter, :]
          W_val = sess.run(W)
          batch_xs_adv = np.zeros([batch_size, 784], dtype=np.float64)
          for s in range(batch_size):
            batch_xs_adv[s, :] = gen_adversary_ex(np.transpose(W_val),
                                            np.transpose(batch_xs[s, :]),
                                            np.argmax(batch_ys[s, :]), eps)
          sess.run(train_step, feed_dict={x: batch_xs_adv, y_true: batch_ys,
                                          lmda : lmda_val})
          
          if t == num_step - 1:
            print('Training finishes!')
            W_val = sess.run(W)
  
            # Adversarial train error on whole dataset
            print('Check adversarial train accuracy...')

            (num_train_all, image_size) = x_train.shape
            x_train_adv = np.zeros([num_train_all, image_size],
                                   dtype=np.float64)
            for i in range(num_train_all):
              x_train_adv[i, :] = gen_adversary_ex(np.transpose(W_val),
                                             np.transpose(x_train[i, :]),
                                             np.argmax(y_train[i, :]), eps)
            each_train_acc = sess.run(accuracy, feed_dict={x: x_train_adv,
                                                y_true: y_train})
            train_acc[eps_idx, lambda_idx, r] = each_train_acc
            print(each_train_acc)
  
            # Adversarial test error on whole dataset
            print('Check adversarial test accuracy...')
            x_test = mnist.test.images
            y_test = mnist.test.labels
            (num_test_all, image_size) = x_test.shape
            x_test_adv = np.zeros([num_test_all, image_size], dtype=np.float64)
            for i in range(num_test_all):
              x_test_adv[i, :] = gen_adversary_ex(np.transpose(W_val),
                                            np.transpose(x_test[i, :]),
                                            np.argmax(y_test[i, :]), eps)
            each_test_acc = sess.run(accuracy, feed_dict={x: x_test_adv,
                                                          y_true: y_test})
            test_acc[eps_idx, lambda_idx, r] = each_test_acc
            print(each_test_acc)
        
        
        np.save('train_acc', train_acc)
        np.save('test_acc', test_acc)
        

  gen_error = train_acc - test_acc
  print('Final train acc')
  print(train_acc)
  print('Final test acc')
  print(test_acc)
  print('Final generalization error')
  print(gen_error)
  np.save('train_acc', train_acc)
  np.save('test_acc', test_acc)
  np.save('gen_error', gen_error)
  print('------------------------------------------------------')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)