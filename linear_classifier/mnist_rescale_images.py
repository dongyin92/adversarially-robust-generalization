from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np

FLAGS = None

def rescale_batch(image_batch, increase=False):
  (num_images, img_size) = image_batch.shape
  # The image has to be a square.
  img_width = int(np.sqrt(img_size))
  assert(img_width * img_width == img_size)
  
  if increase:
    d = int(img_size * 4)
  else:
    assert(img_size % 4 == 0)
    d = int(img_size / 4)
  
  rescale_img_batch = np.zeros([num_images, d], dtype=np.float64)
  
  for i in range(num_images):
    rescale_img_batch[i, :] = rescale(image_batch[i, :].reshape(img_width,
                                                                img_width),
                                      increase).reshape(d)
  
  return rescale_img_batch

def rescale(image, increase=False):
  (d1, d2) = image.shape
  if not increase:
    assert(d1 % 2 == 0 and d2 % 2 == 0)
    new_img = np.zeros([int(d1/2), int(d2/2)], dtype=np.float64)
    for i in range(0, d1, 2):
      for j in range(0, d2, 2):
        new_img[int(i/2), int(j/2)] = np.sqrt(sum(np.square([image[i, j], 
                                                             image[i+1, j],
                                                             image[i, j+1],
                                                             image[i+1, j+1]])))
  else:
    new_img = np.zeros([2 * d1, 2 * d2], dtype=np.float64)
    for i in range(2 * d1):
      for j in range(2 * d2):
        new_img[i, j] = image[int(i/2), int(j/2)]/2.0
  return new_img

def main(_):
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
  mnist_train_labels = mnist.train.labels
  mnist_test_labels = mnist.test.labels
  mnist_train_images_784 = mnist.train.images
  mnist_train_images_196 = rescale_batch(mnist_train_images_784, increase=False)
  mnist_train_images_3136 = rescale_batch(mnist_train_images_784, increase=True)

  mnist_test_images_784 = mnist.test.images
  mnist_test_images_196 = rescale_batch(mnist_test_images_784, increase=False)
  mnist_test_images_3136 = rescale_batch(mnist_test_images_784, increase=True)
  
  np.save('mnist_train_images_784', mnist_train_images_784)
  np.save('mnist_train_images_196', mnist_train_images_196)
  np.save('mnist_train_images_3136', mnist_train_images_3136)

  np.save('mnist_test_images_784', mnist_test_images_784)
  np.save('mnist_test_images_196', mnist_test_images_196)
  np.save('mnist_test_images_3136', mnist_test_images_3136)
  
  np.save('mnist_train_labels', mnist_train_labels)
  np.save('mnist_test_labels', mnist_test_labels)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)