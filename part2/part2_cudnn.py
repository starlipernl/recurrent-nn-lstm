import tensorflow as tf
import numpy as np
import collections
import os
import argparse
import datetime as dt
import matplotlib.pyplot as plt


data_path = "C:/Users/starl/PycharmProjects/ece542_project5/data"

parser = argparse.ArgumentParser()
parser.add_argument('--train', type=int, default=1, help='An integer: 1 to train, 2 for valid, 3 to test')
parser.add_argument('--path', type=str, default=data_path, help='The full path of the training data')
parser.add_argument('--data', type=int, default=1, help='An integer: 1 to train using ptb.train, 2 to train using ptb.char')
args = parser.parse_args()

def read_words(filename):
    with tf.gfile.GFile(filename, "r") as f:
        return f.read().replace("\n", "<eos>").split()


def build_vocab(filename):
    data = read_words(filename)

    count = collections.Counter(data)
    count_pairs = sorted(count.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    keys = dict(zip(words, range(len(words))))

    return keys


def file_to_keys(filename, keys):
    data = read_words(filename)
    return [keys[word] for word in data if word in keys]


def load_data():
    if args.data == 2:
      train_path = os.path.join(data_path, "ptb.char.train.txt")
      valid_path = os.path.join(data_path, "ptb.char.valid.txt")
      test_path = os.path.join(data_path, "ptb.char.test.txt")
    else:  
      train_path = os.path.join(data_path, "ptb.train.txt")
      valid_path = os.path.join(data_path, "ptb.valid.txt")
      test_path = os.path.join(data_path, "ptb.test.txt")

    keys = build_vocab(train_path)
    train_data = file_to_keys(train_path, keys)
    valid_data = file_to_keys(valid_path, keys)
    test_data = file_to_keys(test_path, keys)
    vocabulary = len(keys)
    reversed_dictionary = dict(zip(keys.values(), keys.keys()))

    print ("Data set is loaded")
    return train_data, valid_data, test_data, vocabulary, reversed_dictionary


def batch_producer(raw_data, batch_size, num_steps):
    raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

    data_len = tf.size(raw_data)
    batch_len = data_len // batch_size
    data = tf.reshape(raw_data[0: batch_size * batch_len],
                      [batch_size, batch_len])

    epoch_size = (batch_len - 1) // num_steps

    index = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
    x = data[:, index * num_steps:(index + 1) * num_steps]
    x.set_shape([batch_size, num_steps])
    y = data[:, index * num_steps + 1: (index + 1) * num_steps + 1]
    y.set_shape([batch_size, num_steps])
    return x, y


class Input(object):
    def __init__(self, batch_size, num_steps, data):
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        self.input_data, self.targets = batch_producer(data, batch_size, num_steps)


class Model(object):
    def __init__(self, input, is_training, hidden_size, vocab_size, num_layers,
                 dropout=0.5, scale=0.05):
        self.is_training = is_training
        self.input_obj = input
        self.batch_size = input.batch_size
        self.num_steps = input.num_steps
        self.hidden_size = hidden_size

        with tf.device("/device:GPU:0"):
            embedding = tf.Variable(tf.random_uniform([vocab_size, self.hidden_size], -scale, scale))
            inputs = tf.nn.embedding_lookup(embedding, self.input_obj.input_data)

        if is_training and dropout < 1:
            inputs = tf.nn.dropout(inputs, dropout)

        inputs = tf.transpose(inputs, [1, 0, 2])
        cell = tf.contrib.cudnn_rnn.CudnnLSTM(
            num_layers=num_layers,
            num_units=self.hidden_size,
            #input_size=self.hidden_size,
            dropout=1 - dropout if is_training else 0)
        #params_size_t = cell.count_params()
        # rnn_params = tf.get_variable(
        #     "lstm_params",
        #     initializer=tf.random_uniform(
        #         [params_size_t], -scale, scale),
        #     validate_shape=False)
        c = tf.zeros([num_layers, self.batch_size, self.hidden_size],
                     tf.float32)
        h = tf.zeros([num_layers, self.batch_size, self.hidden_size],
                     tf.float32)
        self.init_state = (tf.contrib.rnn.LSTMStateTuple(h=h, c=c),)
        outputs, (h, c) = cell(inputs)
        #outputs, h, c = cell(inputs, h, c, rnn_params, is_training)
        outputs = tf.transpose(outputs, [1, 0, 2])
        output = tf.reshape(outputs, [-1, self.hidden_size])
        self.state = (tf.contrib.rnn.LSTMStateTuple(h=h, c=c),)

        #self.init_state = tf.placeholder(tf.float32, [num_layers, 2, self.batch_size, self.hidden_size])

        # state_per_layer_list = tf.unstack(self.init_state, axis=0)
        # rnn_tuple_state = tuple(
        #     [tf.contrib.rnn.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1])
        #      for idx in range(num_layers)]
        # )
        #
        # # create an LSTM cell to be unrolled
        # cell = tf.contrib.rnn.LSTMCell(hidden_size, forget_bias=1.0)
        #
        # if is_training and dropout < 1:
        #     cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout)
        # if num_layers > 1:
        #     cell = tf.contrib.rnn.MultiRNNCell([cell for _ in range(num_layers)], state_is_tuple=True)
        #
        # output, self.state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32, initial_state=rnn_tuple_state)
        # # reshape to (batch_size * num_steps, hidden_size)
        # output = tf.reshape(output, [-1, hidden_size])

        softmax_w = tf.Variable(tf.random_uniform([hidden_size, vocab_size], -scale, scale))
        softmax_b = tf.Variable(tf.random_uniform([vocab_size], -scale, scale))
        logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
        # Reshape logits to be a 3-D tensor for sequence loss
        logits = tf.reshape(logits, [self.batch_size, self.num_steps, vocab_size])

        loss = tf.contrib.seq2seq.sequence_loss(
            logits,
            self.input_obj.targets,
            tf.ones([self.batch_size, self.num_steps], dtype=tf.float32),
            average_across_timesteps=False,
            average_across_batch=True)

        # Update the cost
        self.cost = tf.reduce_sum(loss)

        # get the prediction accuracy
        self.softmax_out = tf.nn.softmax(tf.reshape(logits, [-1, vocab_size]))
        self.predict = tf.cast(tf.argmax(self.softmax_out, axis=1), tf.int32)
        correct_prediction = tf.equal(self.predict, tf.reshape(self.input_obj.targets, [-1]))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        if not is_training:
           return
        self.learning_rate = tf.Variable(0.0, trainable=False)

        train_vars = tf.trainable_variables()
        gradient, _ = tf.clip_by_global_norm(tf.gradients(self.cost, train_vars), 5)
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        # optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.apply_gradients(
            zip(gradient, train_vars),
            global_step=tf.contrib.framework.get_or_create_global_step())
        # self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)

        self.new_lr = tf.placeholder(tf.float32, shape=[])
        self.lr_update = tf.assign(self.learning_rate, self.new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self.lr_update, feed_dict={self.new_lr: lr_value})


def train(train_data, vocabulary, num_layers, num_epochs, batch_size, hidden_size,
          learning_rate=1.0, max_lr_epoch=10, lr_decay=0.91, print_iter=100):
    # setup data and models
    training_input = Input(batch_size=batch_size, num_steps=35, data=train_data)
    valid_input = Input(batch_size=batch_size, num_steps=35, data=train_data)
    m = Model(training_input, is_training=True, hidden_size=hidden_size, vocab_size=vocabulary,
              num_layers=num_layers)
    mvalid = Model(valid_input, is_training=False, hidden_size=hidden_size, vocab_size=vocabulary,
              num_layers=num_layers)
    init_op = tf.global_variables_initializer()
    orig_decay = lr_decay
    with tf.Session() as sess:
        # start threads
        if args.data == 1:
            sess.run([init_op])
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        saver = tf.train.Saver()
        perp = np.zeros(num_epochs)
        costs = 0
        iters = 0
        for epoch in range(num_epochs):
            if args.data == 1:
                new_lr_decay = orig_decay ** max(epoch + 1 - max_lr_epoch, 0.0)
                m.assign_lr(sess, learning_rate * new_lr_decay)
                # m.assign_lr(sess, learning_rate)
                current_state = sess.run(m.init_state)
            if args.data == 2:
                current_state = sess.run(mvalid.init_state)
                saver.restore(sess, data_path + '/' + 'md-' + '%d' % epoch)
                for step in range(valid_input.epoch_size):
                    # cost, _ = sess.run([m.cost, m.optimizer])
                    feed_dict = {}
                    for i, (c, h) in enumerate(mvalid.init_state):
                        feed_dict[c] = current_state[i].c
                        feed_dict[h] = current_state[i].h
                    if step % print_iter != 0:
                        cost, current_state = sess.run([mvalid.cost, mvalid.state],
                                                          feed_dict=feed_dict)
                    else:
                        cost, current_state, acc = sess.run([mvalid.cost, mvalid.state, mvalid.accuracy],
                                                               feed_dict=feed_dict)
                        print("Epoch {}, Step {}, cost: {:.3f}, accuracy: {:.3f}".format(epoch,
                                step, cost, acc))
                    costs += cost
                    iters += mvalid.input_obj.num_steps
            else:
                for step in range(training_input.epoch_size):
                    # cost, _ = sess.run([m.cost, m.optimizer])
                    feed_dict = {}
                    for i, (c, h) in enumerate(m.init_state):
                        feed_dict[c] = current_state[i].c
                        feed_dict[h] = current_state[i].h
                    if step % print_iter != 0:
                        cost, _, current_state = sess.run([m.cost, m.train_op, m.state],
                                                          feed_dict=feed_dict)
                    else:
                        cost, _, current_state, acc = sess.run([m.cost, m.train_op, m.state, m.accuracy],
                                                               feed_dict=feed_dict)
                        print("Epoch {}, Step {}, cost: {:.3f}, accuracy: {:.3f}".format(epoch,
                                step, cost, acc))
                    costs += cost
                    iters += m.input_obj.num_steps
            perp[epoch] = np.exp(costs/iters)
            if args.train == 1:
                saver.save(sess, data_path + '/' + 'md-' + '%d' % epoch)

        # do a final save
        if args.train == 1:
            saver.save(sess, data_path + '/' + 'final')
        coord.request_stop()
        coord.join(threads)
        return perp


def test(model_path, test_data, reversed_dictionary, hidden_size):
    test_input = Input(batch_size=20, num_steps=35, data=test_data)
    m = Model(test_input, is_training=False, hidden_size=hidden_size, vocab_size=vocabulary,
              num_layers=2)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # start threads
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        #current_state = np.zeros((2, 2, m.batch_size, m.hidden_size))
        current_state = sess.run(m.init_state)
        # restore the trained model
        saver.restore(sess, model_path)
        # get an average accuracy over num_acc_batches
        num_acc_batches = 30
        # check_batch_idx = [25, 26, 27, 28]
        acc_check_thresh = 5
        accuracy = 0
        pred_string = []
        true_vals_string = []
        for batch in range(num_acc_batches):
            feed_dict = {}
            for i, (c, h) in enumerate(m.init_state):
                feed_dict[c] = current_state[i].c
                feed_dict[h] = current_state[i].h
            true_vals, pred, current_state, acc = sess.run([m.input_obj.targets, m.predict, m.state, m.accuracy],
                                                           feed_dict=feed_dict)
            pred_string = pred_string + [reversed_dictionary[x] for x in pred[:m.num_steps]]
            true_vals_string = true_vals_string + [reversed_dictionary[x] for x in true_vals[0]]
            # print("True values are in the 1st line and predicted values are in 2nd line:")
            # print(" ".join(true_vals_string))
            # print(" ".join(pred_string))
            if batch >= acc_check_thresh:
                accuracy += acc
        print("Prediction average accuracy: {:.3f}".format(accuracy / (num_acc_batches-acc_check_thresh)))
        predictions = ' '.join(pred_string)
        actual_string = ' '.join([str(x) for x in true_vals_string])
        # close threads
        print(predictions)
        print(actual_string)
        coord.request_stop()
        coord.join(threads)

        with open("results.txt", "w") as output:
            for item in actual_string:
                output.write("%s" % item)
            output.write("\n \n")
            for item in predictions:
                output.write("%s" % item)


if args.path:
    data_path = args.path 
train_data, valid_data, test_data, vocabulary, reversed_dictionary = load_data()
num_ep = 10
if args.data == 1:
    hidden = 500
else:
    hidden = 200
if args.train == 1:
    epochs = range(1,num_ep+1)
    perplexity = train(train_data, vocabulary, num_layers=2, num_epochs=num_ep, batch_size=50, hidden_size=hidden)
    #perplexity = perplexity.flatten()
    training_steps = range(1, np.size(perplexity, 0)+1)
    # Loss curves
    plt.figure(1)
    plt.plot(training_steps, perplexity, "r", label="Training Perplexity")
    plt.title("Training Perplexity")
    plt.xlabel("Training Step")
    plt.ylabel("Perplexity")
    plt.savefig('training_perplexity_word_new.png')
elif args.train == 2:
    perplexity = train(valid_data, vocabulary, num_layers=2, num_epochs=num_ep, batch_size=50, hidden_size=hidden)
    #perplexity = perplexity.flatten()
    training_steps = range(1, np.size(perplexity, 0)+1)
    # Loss curves
    plt.figure(1)
    plt.plot(training_steps, perplexity, "b", label="Validation Perplexity")
    plt.title("Validation Perplexity")
    plt.xlabel("Training Step")
    plt.ylabel("Perplexity")
    plt.savefig('valid_perplexity_word_new.png')
else:
    trained_model = args.path + "/final"
    test(trained_model, test_data, reversed_dictionary, hidden)
