import numpy as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight

import warnings
warnings.filterwarnings('ignore')

import openpyxl
from pathlib import Path

epcir = pd.ExcelFile("epc-ir.xlsx", engine='openpyxl')

Triptans = epcir.parse("Triptans")
Triptans_Abstracts = Triptans["abstract"].tolist()
Triptans_Labels = Triptans["Included"]
Triptans_BinaryLabels = pd.factorize(Triptans_Labels)[0]



list_of_tokenized = list()
temp = list()
vocab = Counter()

def save_list(lines, filename):
    data = "\n".join(lines)
    file = open(filename, "w")
    file.write(data)
    file.close()

def load_doc(filename):
    file = open(filename, "r")
    text = file.read()
    file.close()
    return text



def Tokenized(abstracts):
    #tokenized_abstracts = [word_tokenize(str(a)) for a in abstracts]
    Rege = RegexpTokenizer(r"\w+")
    tokenized_abstracts = [Rege.tokenize(str(a)) for a in abstracts]
    for a in tokenized_abstracts:
        tokenized_abstracts = [w.lower() for w in a]
        temp.append(tokenized_abstracts)
    for a in temp:
        tokenized_abstracts = [word for word in a if word.isalpha()]
        list_of_tokenized.append(tokenized_abstracts)

    temp.clear()
    stop_words = set(stopwords.words("english"))

    for a in list_of_tokenized:
        tokenized_abstracts = [w for w in a if not w in stop_words]
        vocab.update(tokenized_abstracts)
        temp.append(tokenized_abstracts)

    return temp


def Untokenize(tokens,vocab):
    untokenized_abstracts = list()
    temp2 = list()
    for a in tokens:
        tkns = [w for w in a if w in vocab]
        temp2.append(tkns)
    for a in temp2:
        text = " ".join(a)
        untokenized_abstracts.append(text)

    return untokenized_abstracts

def encode(untokenized):
    max_length = max([len(s.split()) for s in untokenized])

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(untokenized)
    vocab_size = len(tokenizer.word_index) + 1

    encoded = tokenizer.texts_to_sequences(untokenized)
    padded = pad_sequences(encoded, maxlen=max_length, padding="post")

    return padded, vocab_size, max_length

def split_TestandTrain(padded):


    x_train, x_test, y_train, y_test = train_test_split(padded, Triptans_BinaryLabels, test_size=0.50, random_state=4, stratify=Triptans_BinaryLabels)
    print("XTRAIN SHAPE")
    print(x_train.shape)
    print("X_TEST SHAPE")
    print(x_test.shape)
    return x_train, x_test, y_train, y_test


def define_model(vocab_size, max_length):
    model = Sequential()
    model.add(Embedding(vocab_size, 100, input_length=max_length))
    model.add(Conv1D(filters=100, kernel_size=8, activation="relu"))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(10, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()
    return model



# Save vocab
# tkns = [k for k, c in vocab.items() if c >= 2]
# save_list(tkns, "Tripans_vocab.txt")

vocab_filename = "Tripans_vocab.txt"
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)
tokens = Tokenized(Triptans_Abstracts)


untokenized = Untokenize(tokens, vocab)
padded, vocab_size, max_length = encode(untokenized)
x_train, x_test, y_train, y_test = split_TestandTrain(padded)

model = define_model(vocab_size, max_length)
class_weights = dict(zip(np.unique(Triptans_BinaryLabels), class_weight.compute_class_weight('balanced',np.unique(Triptans_BinaryLabels),Triptans_BinaryLabels)))

model.fit(x_train, y_train, epochs=10, verbose=0,class_weight=class_weights)

_, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Accuracy: {acc}")


# cm = confusion_matrix(y_test, np.argmax(model.predict(x_test),axis=1))
# print(cm)


# y_pred = model.predict(x_test)
# res = np.argmax(y_pred,axis=1)
# print(res)
# # print(classification_report(y_test, np.argmax(model.predict(x_test)), zero_division=0))
# # print(y_pred)
# # percent_pos = y_pred[1,0]
# # print(round(percent_pos))
# c = 0
# for y in y_pred:
#     print(f"y_test:{y_test[c]} -> y_pred:{round(y_pred[c,0])}")
#     c += 1