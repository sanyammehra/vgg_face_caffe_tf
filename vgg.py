import numpy as np
import tensorflow as tf
import time
import utils


class Vgg16:
    def __init__(self, pretrained='./vgg_face_caffe/vgg16.npy', sess=None, dropout=1.0):
        self.params     = {}
        self.features   = {}
        self.X          = tf.placeholder(tf.float32, [None, 224, 224, 3])
        self.keep_probs = dropout
        self.layers = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2',
                        'conv3_1', 'conv3_2', 'conv3_3',
                        'conv4_1', 'conv4_2', 'conv4_3',
                        'conv5_1', 'conv5_2', 'conv5_3',
                        'fc6', 'fc7']
        self._make_layers()
        self.pretrained = pretrained
        self.saver = tf.train.Saver()
        # if self.pretrained and sess is not None:
        #     self._load_weights(self.pretrained, sess)


    def _make_layers(self):
        """
        Builds the graph
        """
        """
        # subtract mean - Imagenet?
        with tf.variable_scope('pre_process') as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            self.X = self.X - mean
        """
        """
        # conv1
        """
        with tf.variable_scope('conv1_1'):
            conv1_1_relu = self._conv_relu(name='conv1_1',
                            inp=self.X,
                            kernel_shape=[3, 3, 3, 64],
                            bias_shape=[64],
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        with tf.variable_scope('conv1_2'):
            conv1_2_relu = self._conv_relu(name='conv1_2',
                            inp=conv1_1_relu,
                            kernel_shape=[3, 3, 64, 64],
                            bias_shape=[64],
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        """
        # pool1
        """
        pool1 = tf.nn.max_pool(conv1_2_relu,
                        ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME',
                        name='pool1')
        """
        # conv2
        """
        with tf.variable_scope('conv2_1'):
            conv2_1_relu = self._conv_relu(name='conv2_1',
                            inp=pool1,
                            kernel_shape=[3, 3, 64, 128],
                            bias_shape=[128],
                            strides=[1, 1, 1, 1],
                            padding='SAME')

        with tf.variable_scope('conv2_2'):
            conv2_2_relu = self._conv_relu(name='conv2_2',
                            inp=conv2_1_relu,
                            kernel_shape=[3, 3, 128, 128],
                            bias_shape=[128],
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        """
        # pool2
        """
        pool2 = tf.nn.max_pool(conv2_2_relu,
                        ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME',
                        name='pool2')
        """
        # conv3
        """
        with tf.variable_scope('conv3_1'):
            conv3_1_relu = self._conv_relu(name='conv3_1',
                            inp=pool2,
                            kernel_shape=[3, 3, 128, 256],
                            bias_shape=[256],
                            strides=[1, 1, 1, 1],
                            padding='SAME')

        with tf.variable_scope('conv3_2'):
            conv3_2_relu = self._conv_relu(name='conv3_2',
                            inp=conv3_1_relu,
                            kernel_shape=[3, 3, 256, 256],
                            bias_shape=[256],
                            strides=[1, 1, 1, 1],
                            padding='SAME')

        with tf.variable_scope('conv3_3'):
            conv3_3_relu = self._conv_relu(name='conv3_3',
                            inp=conv3_2_relu,
                            kernel_shape=[3, 3, 256, 256],
                            bias_shape=[256],
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        """
        # pool3
        """
        pool3 = tf.nn.max_pool(conv3_3_relu,
                        ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME',
                        name='pool3')
        """
        # conv4
        """
        with tf.variable_scope('conv4_1'):
            conv4_1_relu = self._conv_relu(name='conv4_1',
                            inp=pool3,
                            kernel_shape=[3, 3, 256, 512],
                            bias_shape=[512],
                            strides=[1, 1, 1, 1],
                            padding='SAME')

        with tf.variable_scope('conv4_2'):
            conv4_2_relu = self._conv_relu(name='conv4_2',
                            inp=conv4_1_relu,
                            kernel_shape=[3, 3, 512, 512],
                            bias_shape=[512],
                            strides=[1, 1, 1, 1],
                            padding='SAME')

        with tf.variable_scope('conv4_3'):
            conv4_3_relu = self._conv_relu(name='conv4_3',
                            inp=conv4_2_relu,
                            kernel_shape=[3, 3, 512, 512],
                            bias_shape=[512],
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        """
        # pool4
        """
        pool4 = tf.nn.max_pool(conv4_3_relu,
                        ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME',
                        name='pool4')
        """
        # conv5
        """
        with tf.variable_scope('conv5_1'):
            conv5_1_relu = self._conv_relu(name='conv5_1',
                            inp=pool4,
                            kernel_shape=[3, 3, 512, 512],
                            bias_shape=[512],
                            strides=[1, 1, 1, 1],
                            padding='SAME')

        with tf.variable_scope('conv5_2'):
            conv5_2_relu = self._conv_relu(name='conv5_2',
                            inp=conv5_1_relu,
                            kernel_shape=[3, 3, 512, 512],
                            bias_shape=[512],
                            strides=[1, 1, 1, 1],
                            padding='SAME')

        with tf.variable_scope('conv5_3'):
            conv5_3_relu = self._conv_relu(name='conv5_3',
                            inp=conv5_2_relu,
                            kernel_shape=[3, 3, 512, 512],
                            bias_shape=[512],
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        """
        # pool5
        """
        pool5 = tf.nn.max_pool(conv5_3_relu,
                        ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME',
                        name='pool5')
        """
        # fc6
        """
        with tf.variable_scope('fc6'):
            fc6 = self._fc(name='fc6',
                        inp=pool5,
                        weights_shape=[7 * 7 * 512, 4096],
                        bias_shape=[4096])
        relu6 = tf.nn.relu(fc6)
        dropout6 = tf.nn.dropout(relu6, self.keep_probs)
        """
        # fc7
        """
        with tf.variable_scope('fc7'):
            fc7 = self._fc(name='fc7',
                        inp=dropout6,
                        weights_shape=[4096, 4096],
                        bias_shape=[4096])
        """
        relu7 = tf.nn.relu(fc7)
        dropout7 = tf.nn.dropout(relu7, self.keep_probs)
        """
        # fc8
        """
        with tf.variable_scope('fc8'):
            fc8 = _fc(name='fc8',
                        inp=fc7,
                        weights_shape=[4096],
                        bias_shape=[2622])
        """
        # if self.pretrained is not None:
        for k in self.layers:
            assert k in self.features.keys(), '{} was not found in outputs'.format(k)


    def _conv_relu(self, name, inp, kernel_shape, bias_shape, strides=[1, 1, 1, 1], padding='SAME'):
        """
        Implements conv layer followed by Relu
        """
        # initialize weights
        weights = tf.get_variable(name=name + 'weights',
            shape=kernel_shape,
            initializer=tf.contrib.layers.xavier_initializer(),
            trainable=False)
        # initialize biases
        biases = tf.get_variable(name=name + 'biases',
            shape=bias_shape,
            initializer=tf.constant_initializer(0.0),
            trainable=False)
        # convolution
        conv = tf.nn.conv2d(input=inp,
            filter=weights,
            strides=strides,
            padding=padding)
        # record params and features
        self.params[name] = [weights, biases]
        self.features[name] = conv + biases
        return tf.nn.relu(self.features[name])


    def _fc(self, name, inp, weights_shape, bias_shape):
        """
        Implements dense layer
        """
        # initialize weights
        weights = tf.get_variable(name=name + 'weights',
            shape=weights_shape,
            initializer=tf.contrib.layers.xavier_initializer(),
            trainable=False)
        # initialize biases
        biases = tf.get_variable(name=name + 'biases',
            shape=bias_shape,
            initializer=tf.constant_initializer(0.0),
            trainable=False)
        # fcn
        input_flat = tf.reshape(inp, [-1, weights_shape[0]])
        fc = tf.matmul(input_flat, weights) + biases
        # record params and features
        self.params[name] = [weights, biases]
        self.features[name] = fc
        return fc


    def load_weights(self, pretrained_weights, sess):
        """
        Loads pretrained weights into the model
        """
        weights = np.load(pretrained_weights, encoding='latin1').item()
        # for idx, k in enumerate(weights.keys()):
        for idx, k in enumerate(self.layers):
            # print('{} - {} layer with weight: {} ; biases: {}'.format(idx + 1,
            #     k, np.shape(weights[k]['weights']), np.shape(weights[k]['biases'])))
            # assign weights and biases
            sess.run(self.params[k][0].assign(weights[k]['weights']))
            sess.run(self.params[k][1].assign(weights[k]['biases']))
        print('[INFO] Loaded pretrained weights from {}'.format(self.pretrained))


    def extract_features(self, X, sess, batch_size=16):
        # initialize variables and counts
        total_count, feats_fc6, feats_fc7 = 0, {}, {}
        train_indices = np.arange(X.shape[0])
        # randomness not required for feature extraction
        np.random.shuffle(train_indices)
        print('[INFO] Extracting features for {} batches'.format(int(np.ceil(X.shape[0] / batch_size))))
        for mini_batch in range(int(np.ceil(X.shape[0] / batch_size))):
            start = time.time()
            start_idx = (mini_batch * batch_size) % X.shape[0]
            idx = train_indices[start_idx:start_idx + batch_size]
            images = np.array([utils.load_image(p) for p in X[idx]])
            # Get actual batch size
            actual_batch_size = X[mini_batch:mini_batch + batch_size].shape[0]
            total_count += actual_batch_size
            input_feed = {self.X: images}
            output_feed = [self.features]
            # extract and store features
            feats = sess.run(output_feed, input_feed)
            for i, im in enumerate(X[idx]):
                feats_fc6[im] = feats[0]['fc6'][i]
                feats_fc7[im] = feats[0]['fc7'][i]
            end = time.time()
            if mini_batch == 0 or (mini_batch + 1) % 5 == 0:
                print('[INFO] Extracted from mini-batch {} / {} at {:0.2f}s per image'.format(mini_batch + 1, int(np.ceil(X.shape[0] / batch_size)), (end-start)/actual_batch_size))
        return feats_fc6, feats_fc7

    def save_model(self, session, save_path):
        print('[INFO] Saving the model to: {}'.format(save_path))
        self.saver.save(session, save_path)

    def restore_model(self, session, save_path):
        print('[INFO] Restoring the model from: {}'.format(save_path))
        self.saver.restore(session, save_path)
