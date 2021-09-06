from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf

import string
import pickle
import numpy as np
import argparse
import wikipedia

class NWPModel:

    def __init__(self, load_from_file=False, lstm_units=1000, training_data=None, vocab_size=None, x=None, y=None, tokenizer=None, epochs=50):
        self.vocab_size = vocab_size
        self.x = x
        self.y = y
        self.tokenizer = tokenizer
        self.load_from_file = load_from_file
        self.lstm_units = lstm_units
        self.training_data = training_data
        self.epochs = epochs

    def get_wiki_data(self, article):

        print("Getting Wikipedia data for {}".format(article))
        wiki = wikipedia.page(article)
        return wiki.content

    def text_preprocessing(self, train_data=None, wiki=None):

        if wiki:
            data = self.get_wiki_data(wiki)

        else:
            file = open(train_data, "r")
            lines = []

            for i in file:
                lines.append(i)

            data = ""

            for _ in lines:
                data = ' '.join(lines)

        data = data.replace('\n', '').replace('\r', '').replace('\ufeff', '')

        translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
        data = data.translate(translator)

        z = []

        for i in data.split():
            if i not in z:
                z.append(i)

        data = ' '.join(z)

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts([data])

        # saving the tokenizer for predict function.
        pickle.dump(tokenizer, open('tokenizer.pkl', 'wb'))

        sequence_data = tokenizer.texts_to_sequences([data])[0]
        vocab_size = len(tokenizer.word_index) + 1

        sequences = []

        for i in range(1, len(sequence_data)):
            words = sequence_data[i - 1:i + 1]
            sequences.append(words)

        sequences = np.array(sequences)

        X = []
        y = []

        for i in sequences:
            X.append(i[0])
            y.append(i[1])

        X = np.array(X)
        y = np.array(y)

        y = to_categorical(y, num_classes=vocab_size)

        np.savetxt("model_input_x.txt", X)
        np.savetxt("model_input_y.txt", y)

    def load_data(self):

        tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))
        vocab_size = len(tokenizer.word_index) + 1
        x = np.loadtxt("model_input_x.txt")
        y = np.loadtxt("model_input_y.txt")

        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.x = x
        self.y = y

    def build_model(self, print_summary=False):

        self.model = Sequential()
        self.model.add(Embedding(self.vocab_size, 10, input_length=1))
        self.model.add(LSTM(10, return_sequences=True))
        self.model.add(LSTM(self.lstm_units))
        self.model.add(Dense(1000, activation="relu"))
        self.model.add(Dense(self.vocab_size, activation="softmax"))

        if print_summary:
            print(self.model.summary())

    def train_model(self):
        checkpoint = ModelCheckpoint("nextword.h5", monitor='loss', verbose=1,
                                     save_best_only=True, mode='auto')

        reduce = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=3, min_lr=0.0001, verbose=1)
        logdir = 'logsnextword1'

        tensorboard_Visualization = TensorBoard(log_dir=logdir)
        self.model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.001))
        self.model.fit(self.x, self.y, epochs=self.epochs, batch_size=64, callbacks=[checkpoint, reduce, tensorboard_Visualization])

    def predict(self, predict_text):

        use_model = load_model('nextword.h5')
        sequence = np.array(self.tokenizer.texts_to_sequences([predict_text])[0])

        preds = tf.math.top_k(use_model.predict(sequence), 3)

        rev_tokenizer = {}

        for key, value in self.tokenizer.word_index.items():
            rev_tokenizer[value] = key
        # default value for 0 token
        rev_tokenizer[0] = "OTHER"

        predicted_words = list(set([rev_tokenizer[i] for i in preds.indices.numpy().tolist()[0]]))

        return predicted_words


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--train_model', metavar='train_model', type=bool,
                        required=False, default=False, nargs='?', const=True,
                        help='whether to train the model on new text')

    parser.add_argument('--input_text', metavar='input_text', type=str,
                        required=False,
                        help='text used to train the model')

    parser.add_argument('--epochs', metavar='epochs', type=int,
                        required=False, default=50,
                        help='training epochs')

    parser.add_argument('--wiki', metavar='wiki', type=str,
                        required=False, default="",
                        help='wiki article to use for training')

    parser.add_argument('--predict', metavar='predict', type=str,
                        required=False, default="",
                        help='sentence to predict the next word for')

    return parser.parse_args()


def main():

    args = parse_args()

    if args.epochs:
        epochs = args.epochs

    nw_model = NWPModel(epochs = epochs)

    if args.input_text:
        train_data = args.input_text
    else:
        train_data = None

    if args.train_model:
        nw_model.text_preprocessing(train_data=train_data, wiki=args.wiki)
        nw_model.load_data()
        nw_model.build_model()
        nw_model.train_model()

    else:
        nw_model.load_data()

    if args.predict:
        predwords = nw_model.predict(args.predict)
        print("predicged words: (from most likely to least likely")
        print(predwords)

if __name__ == "__main__":
    main()