import numpy as np
import pandas as pd
import re
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
from matplotlib import pyplot


# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


import openpyxl
from pathlib import Path

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


def Tokenized(abstracts):
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

        temp.append(tokenized_abstracts)

    return temp


def Untokenize(tokens):
    untokenized_abstracts = list()

    for a in tokens:
        text = " ".join(a)
        untokenized_abstracts.append(text)

    return untokenized_abstracts


def tf_idf(untokenized):
    tfidf = TfidfVectorizer(use_idf=True)
    vectors = tfidf.fit_transform(untokenized).todense()
    #vectors = tfidf.fit_transform(untokenized)

    # df = pd.DataFrame(vectors[0].T.todense(), index=tfidf.get_feature_names(), columns=["TF-IDF"])
    # df = df.sort_values('TF-IDF', ascending=False)
    # print(df.head(25))

    return vectors



def split_TestandTrain(vector, y):

    x_train, x_test, y_train, y_test = train_test_split(vector, y, test_size=0.7, stratify=y)

    return x_train, x_test, y_train, y_test


def create_model(weight):

    model = LogisticRegression(solver="liblinear", random_state=0, class_weight=weight)

    return model


def run(abstract, label, topic):
    list_of_tokenized.clear()
    temp.clear()

    tokens = Tokenized(abstract)
    untokenized = Untokenize(tokens)
    vectors = tf_idf(untokenized)



    x_train, x_test, y_train, y_test = split_TestandTrain(vectors, label)

    class_weights = dict(zip(np.unique(label),
                             class_weight.compute_class_weight("balanced", np.unique(y_train),
                                                               y_train)))



    #print(f"{topic} weights: {class_weights}")

    model = create_model(class_weights)
    model.fit(x_train, y_train)
    acc = model.score(x_test, y_test)
    #print(f"{topic} Test Accuracy : {acc}")

    cm = confusion_matrix(y_test, model.predict(x_test))
    print(cm)

    R = cm[1,1]/(cm[1,1]+cm[1,0])
    TN = cm[0,0]
    FN = cm[1,0]
    N = cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1]

    WSS = ((TN + FN) / N) - (1- R)

    print(f"WSS for {topic} = {WSS}")



    #print(classification_report(y_test, model.predict(x_test), zero_division=0))

    #y_pred = model.predict(x_test)

    # c = 0
    # for y in y_pred:
    #     print(f"y_test:{y_test[c]} -> y_pred:{y_pred[c]}")
    #     c += 1


run(Triptans_Abstracts, Triptans_BinaryLabels, "Triptans")
run(Statins_Abstracts, Statins_BinaryLabels, "Statins")
run(UI_Abstracts, UI_BinaryLabels, "UrinaryIncontinence")
run(SMR_Abstracts, SMR_BinaryLabels, "SkeletalMuscleRelaxants")
run(Opiods_Abstracts, Opiods_BinaryLabels, "Opiods")
run(OH_Abstracts, OH_BinaryLabels, "OralHypoglycemics")
run(CCB_Abstracts, CCB_BinaryLabels, "CalciumChannelBlockers")
run(PPI_Abstracts, PPI_BinaryLabels, "ProtonPumpInhibitors")
run(Estrogens_Abstracts, Estrogens_BinaryLabels, "Estrogens")
run(NSAIDS_Abstracts, NSAIDS_BinaryLabels, "NSAIDS")
run(BB_Abstracts, BB_BinaryLabels, "BetaBlockers")
run(AH_Abstracts, AH_BinaryLabels, "Antihistamines")
run(AA_Abstracts, AA_BinaryLabels, "AtypicalAntipsychotics")
run(ACEI_Abstracts, ACEI_BinaryLabels, "ACEInhibitors")
run(ADHD_Abstracts, ADHD_BinaryLabels, "ADHD")




# print(classification_report(y_test, model.predict(x_test), zero_division=0))
