from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import tensorflow as tf
import numpy as np
from tensorflow.python import debug as tf_debug
from tensorflow.contrib.rnn import GRUCell
from tensorflow.contrib import rnn
import config


tf.logging.set_verbosity(tf.logging.INFO)

def DeepConvSep_decode(encoded, shape1, shape2, shape3, name = "Decode"):

    encoded_2_1 = tf.layers.dense(encoded, int(shape1[1]) * int(shape1[2]) * int(shape1[3]), name = name+"_1")

    encoded_2_1_re = tf.reshape(encoded_2_1, shape1, name = name+"_2")

    deconv1 = tf.layers.conv2d_transpose(encoded_2_1_re, 50, (15, 1),  name = name+"_3")

    deconv2 = tf.layers.conv2d_transpose(deconv1, 50, (1, 513),  name = name+"_4")

    output = tf.layers.conv2d(deconv2, 1, (1,1), strides=(1,1),  padding = 'valid', name = name+"_5", activation = None)

    # deconv1 = deconv2d(encoded_2_1_re, shape2, k_h = 15,  name = name+"_3")

    # deconv2 = tf.nn.relu(deconv2d(deconv1, shape3, k_w = 513,name = name+"_4"))

    # import pdb;pdb.set_trace()

    return output



def DeepConvSep(input_, feat_size = 513, time_context = 30, num_source = 4):
    
    conv1 = tf.layers.conv2d(input_, 50, (1, feat_size), strides=(1, 1), padding='valid', name="F_1", activation = None)

    conv2 = tf.layers.conv2d(conv1, 50, (int(time_context/2),1), strides=(1,1),  padding = 'valid', name = "F_2", activation = None)

    # import pdb;pdb.set_trace()

    conv2_1 = tf.reshape(conv2, (config.batch_size, -1))

    encoded = tf.layers.dense(conv2_1, 128, name = "encoded")

    outs = []

    for source in range(num_source):
        outs.append(DeepConvSep_decode(encoded, conv2.get_shape(), conv1.get_shape(), input_.get_shape(), name = "Decode"+str(source)))

    output = tf.squeeze(tf.stack(outs, axis = -1))

    return output



def deconv2d(input_, output_shape,
       k_h=1, k_w=1, d_h=1, d_w=1, stddev=0.02,
       name="deconv2d"):

    w = tf.get_variable('w_'+name, [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
              initializer=tf.random_normal_initializer(stddev=stddev))

    # try:
    deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                strides=[1, d_h,d_w, 1], name = name, padding = "SAME")

    biases = tf.get_variable('biases_'+name, [output_shape[-1]], initializer=tf.constant_initializer(0.0))
    deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

    return deconv


def selu(x):
   alpha = 1.6732632423543772848170429916717
   scale = 1.0507009873554804934193349852946
   return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))

def nr_wavenet_block(conditioning, dilation_rate = 2, scope = 'nr_wavenet_block', is_train = False):

    with tf.variable_scope(scope):
        con_pad_forward = tf.pad(conditioning, [[0,0],[dilation_rate,0],[0,0]],"CONSTANT")
        con_pad_backward = tf.pad(conditioning, [[0,0],[0,dilation_rate],[0,0]],"CONSTANT")
        con_sig_forward = tf.layers.batch_normalization(tf.layers.conv1d(con_pad_forward, config.wavenet_filters, 2, dilation_rate = dilation_rate, padding = 'valid'), training=is_train)
        con_sig_backward = tf.layers.batch_normalization(tf.layers.conv1d(con_pad_backward, config.wavenet_filters, 2, dilation_rate = dilation_rate, padding = 'valid'), training=is_train)

        sig = tf.sigmoid(con_sig_forward+con_sig_backward)


        con_tanh_forward = tf.layers.batch_normalization(tf.layers.conv1d(con_pad_forward, config.wavenet_filters, 2, dilation_rate = dilation_rate, padding = 'valid'), training=is_train)
        con_tanh_backward = tf.layers.batch_normalization(tf.layers.conv1d(con_pad_backward, config.wavenet_filters, 2, dilation_rate = dilation_rate, padding = 'valid'), training=is_train)    
        # con_tanh = tf.layers.conv1d(conditioning,config.wavenet_filters,1)

        tanh = tf.tanh(con_tanh_forward+con_tanh_backward)


        outputs = tf.multiply(sig,tanh)

        skip = tf.layers.conv1d(outputs,config.wavenet_filters,1)

        residual = skip + conditioning

    return skip, residual


def nr_wavenet(inputs, f0, is_train):

    input_1 = tf.layers.batch_normalization(tf.layers.dense(inputs, config.wavenet_filters), training=is_train)

    f1 = tf.layers.batch_normalization(tf.layers.dense(f0, config.wavenet_filters), training=is_train)

    prenet_out = tf.concat([input_1, f1], axis = -1)

    num_block = config.wavenet_layers

    receptive_field = 2**num_block

    first_conv = tf.layers.batch_normalization(tf.layers.conv1d(prenet_out, config.wavenet_filters, 1), training=is_train)
    skips = []
    skip, residual = nr_wavenet_block(first_conv, dilation_rate=1, scope = "nr_wavenet_block_0", is_train = is_train)
    output = skip
    for i in range(num_block):
        skip, residual = nr_wavenet_block(residual, dilation_rate=2**(i+1), scope = "nr_wavenet_block_"+str(i+1), is_train = is_train)
        skips.append(skip)
    for skip in skips:
        output+=skip
    output = output+first_conv

    output = tf.nn.relu(output)

    output = tf.layers.batch_normalization(tf.layers.conv1d(output,config.wavenet_filters,1), training=is_train)

    output = tf.nn.relu(output)

    output = tf.layers.batch_normalization(tf.layers.conv1d(output,config.wavenet_filters,1), training=is_train)

    output = tf.nn.relu(output)

    harm = tf.layers.batch_normalization(tf.layers.dense(output, 64, activation=tf.nn.relu), training=is_train)
    # ap = tf.layers.batch_normalization(tf.layers.dense(output, 4, activation=tf.nn.relu), training=is_train)
    # vuv = tf.layers.batch_normalization(tf.layers.dense(ap, 1, activation=tf.nn.sigmoid), training=is_train)

    return harm


if __name__ == '__main__':
    input_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size, config.max_phr_len,513,1),
                                            name='input_placeholder')
    output_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size, config.max_phr_len, 1),
                                             name='output_placeholder')
    haha = DeepConvSep(input_placeholder)