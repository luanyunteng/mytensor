# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A deep MNIST classifier using convolutional layers.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile
import time

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


def deepnn(x):
  """deepnn builds the graph for a deep net for classifying digits.
  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.
  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
  """
  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - there is only one here, since images are
  # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
  with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, 28, 28, 1])

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  # Pooling layer - downsamples by 2X.
  with tf.name_scope('pool1'):
    h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer -- maps 32 feature maps to 64.
  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  # Second pooling layer.
  with tf.name_scope('pool2'):
    h_pool2 = max_pool_2x2(h_conv2)

  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
  # is down to 7x7x64 feature maps -- maps this to 1024 features.
  with tf.name_scope('fc1'):
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Map the 1024 features to 10 classes, one for each digit
  with tf.name_scope('fc2'):
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  return y_conv, keep_prob


def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # Create the model
  x = tf.placeholder(tf.float32, [50, 784])

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [50, 10])

  # Build the graph for the deep net
  y_conv, keep_prob = deepnn(x)

  with tf.name_scope('loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                            logits=y_conv)
  cross_entropy = tf.reduce_mean(cross_entropy)

  with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

  with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)

  graph_location = tempfile.mkdtemp()
  print('Saving graph to: %s' % graph_location)
  train_writer = tf.summary.FileWriter(graph_location)
  train_writer.add_graph(tf.get_default_graph())


  # quanlu: write metagraph
  # tf.add_to_collection("my_accuracy", accuracy)
  # tf.add_to_collection("my_train_step", train_step)
  # tf.add_to_collection("inputs", x)
  # tf.add_to_collection("inputs", y_)
  # tf.add_to_collection("inputs", keep_prob)
  # meta_graph_def = tf.train.export_meta_graph(filename='/tmp/mymodel.meta')

  #tf.reset_default_graph()
  #tf.train.import_meta_graph('/tmp/mymodel.meta')
  #accuracy = tf.get_collection('my_accuracy')[0]
  #train_step = tf.get_collection('my_train_step')[0]
  #[x, y_, keep_prob] = tf.get_collection('inputs')

  config = tf.ConfigProto()
  config.gpu_options.per_process_gpu_memory_fraction = 0.4
  config.graph_options.infer_shapes = True
  with tf.Session(config=config) as sess:
    train_writer = tf.summary.FileWriter('/tmp/mytrain_mnist', sess.graph)
    print('start initialize')
    sess.run(tf.global_variables_initializer())
    print('finish initialize')

    #process the ops' relation
    subg_fd = open('/Users/luanyunteng/github/result/graph_0.txt', "r")
    line = subg_fd.readline()
    dic = {}
    while line:
        split_line = line.split('\t')
        if len(split_line) > 1:
            curID = int(split_line[0].split('[')[1].split(']')[0])
            for ID in split_line[1:]:
                if int(ID) not in dic.keys():
                    dic[int(ID)] = set()
                dic[int(ID)].add(curID)
        line = subg_fd.readline()
    dic[0] = set()
    subg_fd.close()
    #find the critical path
    path = list()
    path.append(1)
    subg_fd = open('/Users/luanyunteng/github/result/graph_0.txt', "r")
    lines = subg_fd.readlines()
    subg_fd.close()
    batch = mnist.train.next_batch(50)
    feed = {x: batch[0], y_: batch[1], keep_prob:0.5}

    current_op_id = 1
    while len(dic[current_op_id]) > 0:
        max_time = -1
        max_time_id = -1
        for op_id in dic[current_op_id]:
            op_id_name = lines[op_id].split('[')[3].split(']')[0]
            op = tf.get_default_graph().get_operation_by_name(op_id_name)
            for i in range(10):
                sess.run(op, feed_dict = feed)
            print("warm up done")
            start = time.clock()
            for i in range(1000):
                sess.run(op, feed_dict = feed)
            res_duration = (time.clock - start) * 1000 / 1000
            print("duration: ", res_duration, "ms")
            if res_duration > max_time:
                max_time = res_duration
                max_time_id = op_id
        print("id: ",max_time_id,"duration: ", max_time, "ms")
        current_op_id = max_time_id
        path.append(current_op_id)
    print(path)
    #
    # #for i in range(10000):
    # for i in range(1):
    #   batch = mnist.train.next_batch(50)
    #   if i % 100 == 0:
    #     print('start accuracy.eval')
    #     train_accuracy = accuracy.eval(feed_dict={
    #         x: batch[0], y_: batch[1], keep_prob: 1.0})
    #     print('step %d, training accuracy %g' % (i, train_accuracy))
    #   #print('batch:', batch[0].shape)
    #   #print('batch_y:', batch[1].shape)
    #   train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    # 
    # #print('test accuracy %g' % accuracy.eval(feed_dict={
    # #    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

