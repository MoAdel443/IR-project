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


# preprocessing
files = natsorted(os.listdir("files"))  # reading files sorted
print(f"Your collection consist of : {files} \n")


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
print("Tokens")
for i in documents:
    # pass
    print(i)


# Stemming
stemmed_documents = []
stemmer = PorterStemmer()
for document in documents:

    stemmed_terms = []
    for term in document:
        stemmed_terms.append(stemmer.stem(term))

    stemmed_documents.append(stemmed_terms)
print("Stemmed Terms")
for i in stemmed_documents:
    print(i)
print("\n")


# positional index

document_number = 1
positional_index = {}

for document in stemmed_documents:
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


print(positional_index, "\n")


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
    for pos, list in enumerate(phrase_list, start=1):  # 1 here bcz doc start from 0 and  enumerate start from 0
        if len(list) == len(q.split()):
            positions.append('doc' + str(pos))
    return positions


# TF and Weighted TF
all_words = []
for document in stemmed_documents:
    for word in document:
        all_words.append(word)


def get_term_frequency(doc):
    # make dict and give val 0 and have no repetition
    words_found = dict.fromkeys(all_words, 0)
    for word in doc:
        words_found[word] += 1
    return words_found


term_freq = pd.DataFrame(get_term_frequency(stemmed_documents[0]).values(), index=get_term_frequency(stemmed_documents[0]).keys())


for i in range(1, len(stemmed_documents)):
    term_freq[i] = get_term_frequency(stemmed_documents[i]).values()

term_freq.columns = ['doc' + str(i) for i in range(1, 11)]

print("term frequency")
print(term_freq, "\n")


def get_weighted_term_freq(x):
    if x > 0:
        return math.log(x) + 1
    else:
        return 0


for i in range(1, len(stemmed_documents) + 1):
    term_freq['doc' + str(i)] = term_freq['doc' + str(i)].apply(get_weighted_term_freq)

print("term frequency")
print(term_freq, "\n")


# Doc Frequency (DF)

doc_freq = pd.DataFrame(columns=['DF', 'IDF'])

for i in range(len(term_freq)):
    frequency = len(positional_index[term_freq.index[i]][1])


    doc_freq.loc[i, 'DF'] = frequency

    n_over_df = (10.0 / frequency)
    doc_freq.loc[i, 'IDF'] = math.log(n_over_df, 10)

doc_freq.index = term_freq.index
print(doc_freq, "\n\n")

tf_idf = term_freq.multiply(doc_freq['IDF'], axis=0)
print("TF.IDF")
print(tf_idf, "\n")

# doc len and normalized tf.idf
document_length = pd.DataFrame()


def get_doc_length(col):
    return np.sqrt(tf_idf[col].apply(lambda x: x ** 2).sum())


for column in tf_idf.columns:
    document_length.loc["length", column + "_length"] = get_doc_length(column)

print(document_length.transpose())


normalized_tf_idf = pd.DataFrame()


def get_normalized_tf_idf(col, x):
    try:
        return x / document_length[col + '_length'].values[0]
    except:
        return 0


for column in tf_idf.columns:
    normalized_tf_idf[column] = tf_idf[column].apply(lambda x: get_normalized_tf_idf(column, x))
print(normalized_tf_idf)


# Query
def rank(q):
    words_found = phrase_query(q)
    if not words_found:
        print("Query doesn't meet any document")
    else:
        query = pd.DataFrame(index=normalized_tf_idf.index)

        query['TF'] = [1 if x in q.split() else 0 for x in list(normalized_tf_idf.index)]

        query["W_TF"] = query['TF'].apply(lambda x: get_weighted_term_freq(x))

        query['IDF'] = doc_freq['IDF']

        query['TF_IDF'] = query['TF'] * query["IDF"]

        query["normalized"] = 0
        for i in range(len(query)):
            query["normalized"].iloc[i] = float(query['TF_IDF'].iloc[i]) / math.sqrt(sum(query['TF_IDF'].values ** 2))

        print(query.loc[q.split()])
        print("\n")
        doc = normalized_tf_idf.multiply(query["normalized"], axis=0)

        scores = {}
        for col in doc.columns:
            if 0 in doc[col].loc[q.split()].values:
                pass
            else:
                scores[col] = doc[col].sum()

        doc_result = doc[list(scores.keys())].loc[q.split()]

        print(doc_result)
        print("\nSUM")
        print(doc_result.sum())

        query_length = math.sqrt(sum([x ** 2 for x in query["TF_IDF"].loc[q.split()]]))

        print("\nQuery Length : " + str(query_length))

        final_result_Sorted = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        print("\nsimilarity")
        print(final_result_Sorted)

        print("\n")
        for doc in final_result_Sorted:
            print(doc[0], end="  ")
        print("\n--------------------")


def boolOp(q1, op, q2):
    x1 = phrase_query(q1)
    x2 = phrase_query(q2)
    if op == "and":
        if q2 == '' or q1 == '':
            print("complete your query!!")
        else:
            res = [x for x in x1 if x in x2]
            print(res)
    elif op == "or":
        if q2 == '' or q1 == '':
            print("complete your query!!")
        else:
            res = x1 + x2
            newRes = list(set(res))
            print(newRes)
    elif op == "andNot":
        if q2 == '' or q1 == '':
            print("complete your query!!")
        else:
            x3 = ['doc1', 'doc2', 'doc3', 'doc4', 'doc5', 'doc6', 'doc7', 'doc8', 'doc9', 'doc10']
            itemsNotInX2 = [item for item in x3 if item not in x2]
            res = [x for x in x1 if x in itemsNotInX2]
            print(res)


def stemmQuery(query):  # take query as input make steem on it and
    query_term = word_tokenize(query)
    stemmed_terms_query = []
    q = ''
    for token in query_term:
        stemmed_terms_query.append(stemmer.stem(token))
    for term in stemmed_terms_query:
        q += f"{term} "
    q = q[:-1]
    return q


# running query at infinite times
flag = True
booleanFlag = False
while flag:
    print("\n1- insert Query\n2- Exit")
    op = input()
    if op == "1":
        query = input("enter query : ")
        query_term = word_tokenize(query)
        for term in query_term:
            if term == 'and' or term == 'or' or term == 'andNot':
                booleanFlag = True
        if booleanFlag is False:  # not bool query
            q = stemmQuery(query)
            rank(q)
            flag = True
        else:
            q1 = ""
            q2 = ""
            op = ''
            for i in range(len(query_term)):
                if query_term[i] == 'and' or query_term[i] == 'or' or query_term[i] == 'andNot':
                    op = query_term[i]
                    x = 0
                    y = i + 1
                    while x < i:
                        q1 += f"{query_term[x]} "
                        x += 1
                    q1 = q1[:-1]
                    while y > i and y < len(query_term):
                        q2 += f"{query_term[y]} "
                        y += 1
                    q2 = q2[:-1]
            q1 = stemmQuery(q1)
            q2 = stemmQuery(q2)
            boolOp(q1, op, q2)
            booleanFlag = False


    else:
        print("Exiting")
        flag = False
