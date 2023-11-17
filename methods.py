import glob
import codecs
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import json

# list of stopwords
stop = set(stopwords.words("english"))

# method to explore the matrix using the matrix and the model
def exploreMatrix(matrix, model):

    # set the terms to the vocabulary of the matrix
    terms = model.vocabulary_

    # get the values of the dictionary
    dict_values = dict(zip(matrix.indices, matrix.data))

    # get the indices of the dictionary
    dict_index = dict(zip(terms.values(), terms.keys()))

    # return the indices and the values
    return dict_index, dict_values


# method to evaluate the model
def evaluateModel(dataSet):

    # declare lists to hold the results
    precision_list = []
    recall_list = []
    fscore_list =[]

    # i is the index
    for i, row, in dataSet.df.iterrows():

        # expected output from the popular terms
        y_pred = topCorpusTerms(dataSet.matrix, dataSet.dict_index, i)

        # the actual keywords
        y_true = dataSet.df.keywords[i]

        # get values from evaluateResults method
        precision, recall, fscore = evaluateResults(y_pred, y_true)

        # append to the lists
        precision_list.append(precision)
        recall_list.append(recall)
        fscore_list.append(fscore)

    # print the results
    print("precision: ", sum(precision_list)/len(precision_list)*100)
    print("recall: ", sum(recall_list)/len(recall_list)*100)
    print("fscore: ", sum(fscore_list)/len(fscore_list)*100)


# method to evaluate the results
def evaluateResults(y_pred, y_true):

    # determine correct
    # populate list with 1 if x is in the true if predicted
    correct = [1 for x in y_true if x in y_pred]

    # sum the number of times correct
    correct = sum(correct)

    # try
    try:

        # calculate the precision
        precision = float(correct/len(y_pred))

    # handle execptions
    except:

        # set precision to 0
        precision = 0

    # try
    try:

        # calculate the recall
        recall = float(correct)/len(y_true)

    # handle execptions
    except:

        # set recall to 0
        recall = 0

    # try
    try:

        # calculate f score
        fscore = 2 * (precision * recall) / (precision + recall)

    # handle execptions
    except:

        # set the f score to 0
        fscore = 0

    # return the precision, recall and fscore
    return precision, recall, fscore
    

# method to clean the text
def cleanText(row):

    # declare a sentence array
    sent = []

    # for each term in the row split them
    for term in row.split():

        # remove everything that is not a lowercase or uppercase letter
        term = re.sub('[^a-zA-Z]', " ", term.lower())

        # append the term to the sentence array
        sent.append(term)

    #split = " ", join(sent)
    join = [word for word in sent if word not in stop]

    # return the formatted setence array
    return " ".join(sent)


# apply tfidf to the data
def applyTFIDF(data):

    # n gram range is used to increase the number of features
    # n gram allows for context of words
    tfidf_vectoriser = TfidfVectorizer(max_df = 0.9, min_df = 0.1, ngram_range = (1,4))

    # set the matrix to the fitted and transformed data
    tfidf_matrix = tfidf_vectoriser.fit_transform(list(data))

    # return both the tfidf_vectoriser, tfidf_matrix
    return tfidf_vectoriser, tfidf_matrix
    

# find the top corpus terms
def topCorpusTerms(matrix, model_terms, row_id, top_n = 10):

    # set the row
    row = np.squeeze(matrix[row_id].toarray())

    # slice up the array
    topn_ids = np.argsort(row)[::-1]

    # find the top terms
    top_terms = [model_terms[i] for i in topn_ids]

    # return it up to 10 terms
    return top_terms[:top_n]


# extracts the file path
def extractFilePath(path):

    # find the keys from the path
    keys = glob.glob(f'{path}//**/*.kwd')

    # find the texts from the path
    texts = glob.glob(f'{path}//**/*.xml')

    # declare the lists for values
    key_paths = []
    text_paths = []
    key_number_list = []

    # get the key number list
    for i in range(len(keys)):
        keypaths = keys[i].split("/")
        value = keypaths[-1].split(".")[0]
        key_number_list.append(value)

    # get the text list
    for i in range(len(texts)):
        textpath = texts[i].split("/")
        value = textpath[-1].split(".")[0]

        if value in key_number_list:
            text_paths.append(texts[i])

    # return sorted keys and text paths
    return sorted(keys), sorted(text_paths)
    

# method to extract the lines of the document
def extractDocLines(path):

    # read out the data
    with codecs.open(path, "r", encoding = "utf8", errors = "ignore") as f:

        #
        doc = f.read()
        doc = doc.split("\n")
        doc = "".join(doc)

        # declare sections array
        sectionsArray = []

        # declare sections text
        sectionText = ""

        # look into the document for section text
        for result in re.findall("<SECTION(.*?)</SECTION>", doc):

            # append the result to the array
            sectionsArray.append(result)

            # add the result to the section text
            sectionText += result

        # obtain headers from section array
        sectionHeaders = []

        # for each section
        for section in sectionsArray:

            # for each result find the header
            for result in re.findall('header=(.*?)>', section):

                # append the section headers to the array
                sectionHeaders.append(result)

    return sectionText

# method to extract the keywords
def extractKeywords(path):

    # open the file
    with codecs.open(path, "r", encoding = "utf8", errors = "ignore") as f:

        # read the document
        doc = f.read()

        # each of the key words saved in a new line
        doc = doc.split("\n")

        # format the document
        doc = [x.lower() for x in doc if x]

    # return the doc
    return doc


# method to improve the ngrams
def bolsterNgrams(dataSet):

    terms = dataSet.dict_index.values()

    # lexemes are compound nouns
    lexemes = []

    # for all of the corpus terms
    for term in terms:

        # if the word is two words combined
        if len(term.split()) > 1:

            # append to the lexemes list
            lexemes.append(term)

    # for each index and row in the
    for i, row in dataSet.df.iterrows():

        # for each compound nounes
        for lex in lexemes:

            # check if the lex is in each document
            if lex in dataSet.df.vsm[i]:

                # count the instances
                count = dataSet.df.vsm[i].count(lex)

                # increment the count
                dataSet.matrix[i, dataSet.model.vocabulary_[lex]] *= count

    return lexemes

'''
# rewritten
def bolster_ngrams(dataSet):

    terms = dataSet.dict_index.values()

    lexemes = []

    for term in terms:

        if len(term.split()) > 1:

            lexemes.append(term)

    for i, row in dataSet.df.iterrows():

        for lex in lexemes:

            if lex in dataSet.df.vsm[i]:

                count = dataSet.df.vsm[i].count(lex)

                dataSet.matrix[i, dataSet.model.vocabulary_[lex]] *= count

    return lexemes

'''
