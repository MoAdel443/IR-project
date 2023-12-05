import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from natsort import natsorted  # to sort files
import pandas as pd
import numpy as np
import math
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None  # hide warning

# create a list of StopWords
stopWords = set(stopwords.words('english'))
stopWords.remove('to')
stopWords.remove('in')
stopWords.remove('where')  # delete these words from stop list

files = natsorted(os.listdir("files"))  # reading files sorted

# print(f"Your collection consist of : {files} \n")


# tokenization
documents = []
for file in files:
    with open(f"files/{file}", "r") as f:
        document = f.read()
    tokens = word_tokenize(document)  # tokenize every document in document collection

    terms = []
    for token in tokens:
        if token not in stopWords:  # removing stopwords
            terms.append(token)
    documents.append(terms)

# print("Tokens")
for i in documents:
    pass
    # print(i)


# Stemming
stemmed_documents = []
stemmer = PorterStemmer()
for document in documents:

    stemmed_terms = []
    for term in document:
        stemmed_terms.append(stemmer.stem(term))

    stemmed_documents.append(stemmed_terms)

# print("Stemmed Terms")
for i in stemmed_documents:
    pass
    # print(i)
# print("\n")


# positional index

document_number = 1
positional_index = {}

for document in documents:

    for position, term in enumerate(document):

        # first time to add term
        if term in positional_index:
            positional_index[term][0] = positional_index[term][0] + 1  # increase frequency by 1

            if document_number in positional_index[term][1]:  # if doc exist append to it
                positional_index[term][1][document_number].append(position)

            else:
                positional_index[term][1][document_number] = [position]

        # term already exist
        else:
            positional_index[term] = []
            positional_index[term].append(1)  # frequency
            positional_index[term].append({})  # for doc and postings
            positional_index[term][1][document_number] = [position]  # 1 bcz 0 is the frequency

    document_number += 1

# print(positional_index, "\n")


# phrase query

def phrase_query(q):
    phrase_list = [[] for i in range(len(files))]  # create 10 empty lists for docs

    for word in q.split():
        if word not in positional_index:
            print("Wrong Query")
        else:

            for key in positional_index[word][1].keys():

                # check if first word and sec exist in any file

                if phrase_list[key - 1] != []:
                    if phrase_list[key - 1][-1] == positional_index[word][1][key][0] - 1:
                        phrase_list[key - 1].append(positional_index[word][1][key][0])
                else:
                    phrase_list[key - 1].append(positional_index[word][1][key][0])
    positions = []
    for pos, list in enumerate(phrase_list, start=1):  # 2 here bcz doc start from 0 and  enumerate start from 0
        if len(list) == len(q.split()):
            positions.append('doc' + str(pos))
    return positions


# TF and Weighted TF

all_words = []
for document in documents:
    for word in document:
        all_words.append(word)


def get_term_frequency(doc):
    # make dict and give val 0 and have no repetition
    words_found = dict.fromkeys(all_words, 0)
    for word in doc:
        words_found[word] += 1
    return words_found


term_freq = pd.DataFrame(get_term_frequency(documents[0]).values(), index=get_term_frequency(documents[0]).keys())

for i in range(1, len(documents)):
    term_freq[i] = get_term_frequency(documents[i]).values()

term_freq.columns = ['doc' + str(i) for i in range(1, 11)]


# print("term frequency")
# print(term_freq, "\n")


def get_weighted_term_freq(x):
    if x > 0:
        return math.log(x) + 1
    else:
        return 0


for i in range(1, len(documents) + 1):
    term_freq['doc' + str(i)] = term_freq['doc' + str(i)].apply(get_weighted_term_freq)

# print("term frequency")
# print(term_freq, "\n")

# Doc Frequency

doc_freq = pd.DataFrame(columns=['DF', 'IDF'])

for i in range(len(term_freq)):
    frequency = term_freq.iloc[i].values.sum()

    doc_freq.loc[i, 'DF'] = frequency
    n_over_df = (10.0 / frequency)
    doc_freq.loc[i, 'IDF'] = math.log(n_over_df, 10)

doc_freq.index = term_freq.index
# print(doc_freq, "\n\n")

tf_idf = term_freq.multiply(doc_freq['IDF'], axis=0)

# print("TF.IDF")
# print(tf_idf, "\n")

# doc len and normalized tf.idf

document_length = pd.DataFrame()


def get_doc_length(col):
    return np.sqrt(tf_idf[col].apply(lambda x: x ** 2).sum())


for column in tf_idf.columns:
    document_length.loc[0, column + "_length"] = get_doc_length(column)

# print(document_length)

normalized_tf_idf = pd.DataFrame()


def get_normalized_tf_idf(col, x):
    try:
        return x / document_length[col + '_length'].values[0]
    except:
        return 0


for column in tf_idf.columns:
    normalized_tf_idf[column] = tf_idf[column].apply(lambda x: get_normalized_tf_idf(column, x))


# print(normalized_tf_idf)


# Query

# q = "fools fear in rush"


def rank(q):
    words_found = phrase_query(q)
    if not words_found:
        print("Query doesn't meet any document")
    else:
        query = pd.DataFrame(index=normalized_tf_idf.index)

        query['TF'] = [1 if x in q.split() else 0 for x in list(normalized_tf_idf.index)]

        query["W_TF"] = query['TF'].apply(lambda x: get_weighted_term_freq(x))

        doc1 = normalized_tf_idf.multiply(query["W_TF"], axis=0)

        query['IDF'] = doc_freq['IDF'] * query['W_TF']

        query['TF_IDF'] = query['TF'] * query["IDF"]

        query["normalized"] = 0
        for i in range(len(query)):
            query["normalized"].iloc[i] = float(query['IDF'].iloc[i]) / math.sqrt(sum(query['IDF'].values ** 2))

        print(query.loc[q.split()])
        print("\n")

        doc2 = doc1.multiply(query["normalized"], axis=0)

        scores = {}
        for col in doc2.columns:
            if 0 in doc2[col].loc[q.split()].values:
                pass
            else:
                scores[col] = doc2[col].sum()

        doc_result = doc2[list(scores.keys())].loc[q.split()]

        print(doc_result)
        print("\nSUM")
        print(doc_result.sum())

        query_length = math.sqrt(sum([x ** 2 for x in query["IDF"].loc[q.split()]]))

        print("\nQuery Length : " + str(query_length))

        final_result_Sorted = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        print("\nsimilarity")
        print(final_result_Sorted)

        print("\n")
        for doc in final_result_Sorted:
            print(doc[0], end="  ")
        print("\n--------------------")


# running query at infinite times
flag = True
while flag:
    print("\n1- insert Query\n2- Exit")
    op = input()
    if op == "1":
        query = input("enter query : ")
        rank(query)
        flag = True
    else:
        print("Exiting")
        flag = False
