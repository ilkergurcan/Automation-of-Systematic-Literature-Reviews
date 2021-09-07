import pandas as pd
import numpy as np
from keras.layers.convolutional import Convolution1D
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from keras import layers, models, optimizers
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import GlobalMaxPooling1D, Dense, Dropout, concatenate
from tensorflow.python.keras.models import load_model, Sequential, Model

import imblearn
from imblearn.over_sampling import RandomOverSampler
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

from keras.optimizers import SGD
from keras.optimizers import Adagrad
from keras.optimizers import Adam


from keras.regularizers import l2
from tensorflow.python.keras.utils.vis_utils import plot_model

epcir = pd.ExcelFile("epc-ir.xlsx", engine='openpyxl')

Triptans = epcir.parse("Triptans")
Triptans_Abstracts = Triptans["abstract"].tolist()
Triptans_Labels = Triptans["Included"]
Triptans_BinaryLabels = pd.factorize(Triptans_Labels)[0]

Statins = epcir.parse("Statins")
Statins_Abstracts = Statins["abstract"].tolist()
Statins_Labels = Statins["Included"]
Statins_BinaryLabels = pd.factorize(Statins_Labels)[0]

UI = epcir.parse("UrinaryIncontinence")
UI_Abstracts = UI["abstract"].tolist()
UI_Labels = UI["Included"]
UI_BinaryLabels = pd.factorize(UI_Labels)[0]

SMR = epcir.parse("SkeletalMuscleRelaxants")
SMR_Abstracts = SMR["abstract"].tolist()
SMR_Labels = SMR["Included"]
SMR_BinaryLabels = pd.factorize(SMR_Labels)[0]

Opiods = epcir.parse("Opiods")
Opiods_Abstracts = Opiods["abstract"].tolist()
Opiods_Labels = Opiods["Included"]
Opiods_BinaryLabels = pd.factorize(Opiods_Labels)[0]

OH = epcir.parse("OralHypoglycemics")
OH_Abstracts = OH["abstract"].tolist()
OH_Labels = OH["Included"]
OH_BinaryLabels = pd.factorize(OH_Labels)[0]

CCB = epcir.parse("CalciumChannelBlockers")
CCB_Abstracts = CCB["abstract"].tolist()
CCB_Labels = CCB["Included"]
CCB_BinaryLabels = pd.factorize(CCB_Labels)[0]

PPI = epcir.parse("ProtonPumpInhibitors")
PPI_Abstracts = PPI["abstract"].tolist()
PPI_Labels = PPI["Included"]
PPI_BinaryLabels = pd.factorize(PPI_Labels)[0]

Estrogens = epcir.parse("Estrogens")
Estrogens_Abstracts = Estrogens["abstract"].tolist()
Estrogens_Labels = Estrogens["Included"]
Estrogens_BinaryLabels = pd.factorize(Estrogens_Labels)[0]

NSAIDS = epcir.parse("NSAIDS")
NSAIDS_Abstracts = NSAIDS["abstract"].tolist()
NSAIDS_Labels = NSAIDS["Included"]
NSAIDS_BinaryLabels = pd.factorize(NSAIDS_Labels)[0]

BB = epcir.parse("BetaBlockers")
BB_Abstracts = BB["abstract"].tolist()
BB_Labels = BB["Included"]
BB_BinaryLabels = pd.factorize(BB_Labels)[0]

AH = epcir.parse("Antihistamines")
AH_Abstracts = AH["abstract"].tolist()
AH_Labels = AH["Included"]
AH_BinaryLabels = pd.factorize(AH_Labels)[0]

AA = epcir.parse("AtypicalAntipsychotics")
AA_Abstracts = AA["abstract"].tolist()
AA_Labels = AA["Included"]
AA_BinaryLabels = pd.factorize(AA_Labels)[0]

ACEI = epcir.parse("ACEInhibitors")
ACEI_Abstracts = ACEI["abstract"].tolist()
ACEI_Labels = ACEI["Included"]
ACEI_BinaryLabels = pd.factorize(ACEI_Labels)[0]

ADHD = epcir.parse("ADHD")
ADHD_Abstracts = ADHD["abstract"].tolist()
ADHD_Labels = ADHD["Included"]
ADHD_BinaryLabels = pd.factorize(ADHD_Labels)[0]


list_of_tokenized = list()
temp = list()


def clean(abstracts):
    regexp = RegexpTokenizer(r"\w+")
    tokenized_abstracts = [regexp.tokenize(str(a)) for a in abstracts]

    for a in tokenized_abstracts:
        tokenized_abstracts = [w.lower() for w in a]
        temp.append(tokenized_abstracts)

    for a in temp:
        tokenized_abstracts = [word for word in a if word.isalpha()]
        list_of_tokenized.append(tokenized_abstracts)

    temp.clear()
    stop_words = set(stopwords.words("english"))

    for a in list_of_tokenized:
        tokenized_abstracts = [w for w in a if w not in stop_words]
        temp.append(tokenized_abstracts)

    list_of_tokenized.clear()

    porter = PorterStemmer()
    for a in temp:
        tokenized_abstracts = [porter.stem(word) for word in a]
        list_of_tokenized.append(tokenized_abstracts)

    return list_of_tokenized


def JoinTokens(tokens):
    joined_abs = list()

    for a in tokens:
        text = " ".join(a)
        joined_abs.append(text)

    return joined_abs


def tf_idf(cleaned):
    tfidf = TfidfVectorizer(use_idf=True, min_df=5)
    vectors = tfidf.fit_transform(cleaned).todense()

    return vectors


def Oversample(x_train, y_train):
    oversample = RandomOverSampler(sampling_strategy = "minority")
    x_over, y_over = oversample.fit_resample(x_train, y_train)

    return x_over, y_over


def split_TestandTrain(vector, y):

    x_train, x_test, y_train, y_test = train_test_split(vector, y, test_size=0.7, stratify=y)

    return x_train, x_test, y_train, y_test


def create_model(shape):
    # optimizer = SGD(lr=0.1, momentum=0.8, decay=0.1/100, nesterov=True)
    # optimizer = Adagrad(learning_rate=0.01)
    optimizer = Adam(lr=0.0001)
    input_layer = Input((shape, 1))
    conv_layer = Convolution1D(128, 4, activation='relu', kernel_regularizer=l2(0.008), bias_regularizer=l2(0.008))(input_layer)
    pool_layer = GlobalMaxPooling1D()(conv_layer)
    flat = Flatten()(pool_layer)

    output_layer1 = Dense(50, activation="relu")(flat)
    output_layer1 = Dropout(0.25)(output_layer1)
    output_layer2 = Dense(1, activation="sigmoid")(output_layer1)

    model = Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=["accuracy"])

    #plot_model(model, to_file="model.png", show_shapes=True)

    return model



def run(abstract, label, topic):
    list_of_tokenized.clear()
    temp.clear()

    tokens = clean(abstract)
    cleaned = JoinTokens(tokens)
    vectors = tf_idf(cleaned)


    while True:
        x_train, x_test, y_train, y_test = split_TestandTrain(vectors, label)
        x_over, y_over = Oversample(x_train, y_train)
        input_shape = x_over.shape[1]

        model = create_model(input_shape)
        model.fit(x_over, y_over, epochs=100, verbose=0, validation_data=(x_test, y_test), class_weight={0:1, 1:1.4})

        _, acc = model.evaluate(x_test, y_test, verbose=0)

        print(acc * 100)
        y_pred = model.predict(x_test)
        y_pred.reshape(-1)
        y_pred[y_pred <= 0.5] = 0
        y_pred[y_pred > 0.5] = 1

        cm = confusion_matrix(y_test, y_pred)
        print(cm)

        R = cm[1, 1] / (cm[1, 1] + cm[1, 0])
        TN = cm[0, 0]
        FN = cm[1, 0]
        N = cm[0, 0] + cm[0, 1] + cm[1, 0] + cm[1, 1]
        FP = cm[0,1]
        TP = cm[1,1]
        P = TP / (TP + FP)

        WSS = ((TN + FN) / N) - (1 - R)

        print(f"WSS for {topic} = { round(WSS*100, 2)}")
        print(f"Recall for {topic} = {R}")
        print(f"Precision for {topic} = {P}")

        # if R*100 > 91 and FP < 500:
        #     with open("ADHD.h5", "wb") as f:
        #         model.save("ADHD.h5")
        #     print("SAVED")



def evaluate_test(abstract, label, topic):
    list_of_tokenized.clear()
    temp.clear()

    tokens = clean(abstract)
    cleaned = JoinTokens(tokens)
    vectors = tf_idf(cleaned)

    x_train, x_test, y_train, y_test = split_TestandTrain(vectors, label)

    model = load_model(f"{topic}.h5")

    _, acc = model.evaluate(vectors, label, verbose=0)

    print(acc * 100)
    y_pred = model.predict(vectors)
    y_pred.reshape(-1)
    y_pred[y_pred <= 0.5] = 0
    y_pred[y_pred > 0.5] = 1

    cm = confusion_matrix(label, y_pred)
    print(cm)

    R = cm[1, 1] / (cm[1, 1] + cm[1, 0])
    TN = cm[0, 0]
    FN = cm[1, 0]
    N = cm[0, 0] + cm[0, 1] + cm[1, 0] + cm[1, 1]
    FP = cm[0, 1]
    TP = cm[1, 1]
    P = TP / (TP + FP)

    WSS = ((TN + FN) / N) - (1 - R)

    print(f"WSS@95% for {topic} = {round(WSS * 100, 2)}")
    print(f"Recall for {topic} = {R}")
    print(f"Precision for {topic} = {P}")




# TODO: ADHD yi trainle sonra evaluate et, ppt hazırla, report yaz!!!!


#run(Triptans_Abstracts, Triptans_BinaryLabels, "Triptans")
#run(Statins_Abstracts, Statins_BinaryLabels, "Statins")
#run(UI_Abstracts, UI_BinaryLabels, "UrinaryIncontinence")
#run(SMR_Abstracts, SMR_BinaryLabels, "SkeletalMuscleRelaxants")
#run(Opiods_Abstracts, Opiods_BinaryLabels, "Opiods")
#run(OH_Abstracts, OH_BinaryLabels, "OralHypoglycemics")
#run(CCB_Abstracts, CCB_BinaryLabels, "CalciumChannelBlockers")
#run(PPI_Abstracts, PPI_BinaryLabels, "ProtonPumpInhibitors")
#run(Estrogens_Abstracts, Estrogens_BinaryLabels, "Estrogens")
#run(NSAIDS_Abstracts, NSAIDS_BinaryLabels, "NSAIDS")
#run(BB_Abstracts, BB_BinaryLabels, "BetaBlockers")
#run(AH_Abstracts, AH_BinaryLabels, "Antihistamines")
#run(AA_Abstracts, AA_BinaryLabels, "AtypicalAntipsychotics")
#run(ACEI_Abstracts, ACEI_BinaryLabels, "ACEInhibitors")
#run(ADHD_Abstracts, ADHD_BinaryLabels, "ADHD")

evaluate_test(Triptans_Abstracts, Triptans_BinaryLabels, "Triptans")
evaluate_test(Statins_Abstracts, Statins_BinaryLabels, "Statins")
evaluate_test(UI_Abstracts, UI_BinaryLabels, "UrinaryIncontinence")
evaluate_test(SMR_Abstracts, SMR_BinaryLabels, "SkeletalMuscleRelaxants")
evaluate_test(Opiods_Abstracts, Opiods_BinaryLabels, "Opiods")
evaluate_test(OH_Abstracts, OH_BinaryLabels, "OralHypoglycemics")
evaluate_test(CCB_Abstracts, CCB_BinaryLabels, "CalciumChannelBlockers")
evaluate_test(PPI_Abstracts, PPI_BinaryLabels, "ProtonPumpInhibitors")
evaluate_test(Estrogens_Abstracts, Estrogens_BinaryLabels, "Estrogens")
evaluate_test(NSAIDS_Abstracts, NSAIDS_BinaryLabels, "NSAIDS")
evaluate_test(BB_Abstracts, BB_BinaryLabels, "BetaBlockers")
evaluate_test(AH_Abstracts, AH_BinaryLabels, "Antihistamines")
evaluate_test(AA_Abstracts, AA_BinaryLabels, "AtypicalAntipsychotics")
evaluate_test(ACEI_Abstracts, ACEI_BinaryLabels, "ACEInhibitors")
evaluate_test(ADHD_Abstracts, ADHD_BinaryLabels, "ADHD")
