import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation, Embedding, Dropout, TimeDistributed
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
import collections
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt

data_path = "C:/Users/starl/PycharmProjects/ece542_project5/data"

parser = argparse.ArgumentParser()
parser.add_argument('--run_opt', type=int, default=1, help='An integer: 1 to train, 2 to test')
parser.add_argument('--dpath', type=str, default=data_path, help='The full path of the training data')
parser.add_argument('--data', type=int, default=1, help='integer: 1 train using ptb.train, 2 train using ptb.char')
args = parser.parse_args()
if args.dpath:
    data_path = args.dpath

def read_words(filename):
    with tf.gfile.GFile(filename, "r") as f:
        return f.read().replace("\n", "<eos>").split()


def build_vocab(filename):
    data = read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id


def file_to_word_ids(filename, word_to_id):
    data = read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]


def load_data():
    if args.data == 2:
      train_path = os.path.join(data_path, "ptb.char.train.txt")
      valid_path = os.path.join(data_path, "ptb.char.valid.txt")
      test_path = os.path.join(data_path, "ptb.char.test.txt")
    else:
      train_path = os.path.join(data_path, "ptb.train.txt")
      valid_path = os.path.join(data_path, "ptb.valid.txt")
      test_path = os.path.join(data_path, "ptb.test.txt")

    # build the complete vocabulary, then convert text data to list of integers
    word_to_id = build_vocab(train_path)
    train_data = file_to_word_ids(train_path, word_to_id)
    valid_data = file_to_word_ids(valid_path, word_to_id)
    test_data = file_to_word_ids(test_path, word_to_id)
    vocab = len(word_to_id)
    reverse_dict = dict(zip(word_to_id.values(), word_to_id.keys()))

    return train_data, valid_data, test_data, vocab, reverse_dict


class Batch_Generator(object):

    def __init__(self, data, num_steps, batch_size, vocab, skip_step=5):
        self.data = data
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.vocab = vocab
        self.current_idx = 0
        self.skip_step = skip_step

    def generate(self):
        x = np.zeros((self.batch_size, self.num_steps))
        y = np.zeros((self.batch_size, self.num_steps, self.vocab))
        while True:
            for i in range(self.batch_size):
                if self.current_idx + self.num_steps >= len(self.data):
                    # reset the index
                    self.current_idx = 0
                x[i, :] = self.data[self.current_idx:self.current_idx + self.num_steps]
                temp_y = self.data[self.current_idx + 1:self.current_idx + self.num_steps + 1]
                # one-hot conversion
                y[i, :, :] = to_categorical(temp_y, num_classes=self.vocab)
                self.current_idx += self.skip_step
            yield x, y


train_data, valid_data, test_data, vocab, reverse_dict = load_data()
num_steps = 35
batch_size = 100
train_generator = Batch_Generator(train_data, num_steps, batch_size, vocab,
                                           skip_step=num_steps)
valid_generator = Batch_Generator(valid_data, num_steps, batch_size, vocab,
                                           skip_step=num_steps)
if args.data == 2:
    hidden_size = 150
else:
    hidden_size = 500
use_dropout=True
model = Sequential()
model.add(Embedding(vocab, hidden_size, input_length=num_steps))
model.add(LSTM(hidden_size, return_sequences=True))
model.add(LSTM(hidden_size, return_sequences=True))
if use_dropout:
    model.add(Dropout(0.5))
model.add(TimeDistributed(Dense(vocab)))
model.add(Activation('softmax'))
optimizer = Adam()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

checkpointer = ModelCheckpoint(filepath=data_path + '/model-{epoch:02d}_word.hdf5', verbose=1)
num_epochs = 10
if args.run_opt == 1:
    history = model.fit_generator(train_generator.generate(), len(train_data)//(batch_size*num_steps), num_epochs,
                        validation_data=valid_generator.generate(),
                        validation_steps=len(valid_data)//(batch_size*num_steps), callbacks=[checkpointer])
    history_dict = history.history
    # train_accuracy = history_dict["acc"]
    train_loss = history_dict["loss"]
    # valid_accuracy = history_dict["val_acc"]
    valid_loss = history_dict["val_loss"]
    train_perplexity = np.exp(train_loss)
    valid_perplexity = np.exp(valid_loss)
    model.save(data_path + "/final_model.hdf5")
    plt.figure(1)
    plt.plot(range(1, num_epochs + 1), train_perplexity, "r", label="Training Perplexity")
    plt.title("Training Perplexity")
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    plt.savefig('training_perplexity_word_new.png')
    plt.figure(2)
    plt.plot(range(1, num_epochs + 1), valid_perplexity, "r", label="Training Perplexity")
    plt.title("Validation Perplexity")
    plt.xlabel("Epoch")
    plt.ylabel("Validation")
    plt.savefig('valid_perplexity_word_new.png')
elif args.run_opt == 2:
    model = load_model(data_path + "/final_model.hdf5")
    skip_iters = 50
    # test data set
    example_generator = Batch_Generator(test_data, num_steps, 1, vocab,
                                                     skip_step=1)
    print("Test data:")
    for i in range(skip_iters):
        dummy = next(example_generator.generate())
    if args.data == 2:
        num_predict = 500
    else:
        num_predict = 100
    true_val_out = "Actual words:    "
    pred_out = "Predicted words: "
    for i in range(num_predict):
        data = next(example_generator.generate())
        predictions = model.predict(data[0])
        predict_word = np.argmax(predictions[:, num_steps - 1, :])
        if args.data == 1:
            true_val_out += reverse_dict[test_data[num_steps + skip_iters + i]] + " "
            pred_out += reverse_dict[predict_word] + " "
        else:
            true_val_out += reverse_dict[test_data[num_steps + skip_iters + i]]
            pred_out += reverse_dict[predict_word]
    print(true_val_out)
    print(pred_out)
    with open("results_new.txt", "w") as output:
        for item in true_val_out:
            output.write("%s" % item)
        output.write("\n")
        for item in pred_out:
            output.write("%s" % item)

