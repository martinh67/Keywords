# import statements required to run the program
import time
import glob
from dataClass import DataSet
from methods import *
import codecs
import re
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

# first create argparser

# https://github.com/snkim/AutomaticKeyphraseExtraction

# allows us to run things on the command line
import argparse

# start the timer
start = time.time()

# create argument parser object
parser = argparse.ArgumentParser()

# add f argument to the parser
parser.add_argument("-f", required = False, help = "please specify filepath")

# add the g argument to the parser
parser.add_argument("-g", required = False, help = "run target function")

# display the args
args = parser.parse_args()

# declare the root of the data
root = "/Users/martinhanna/Downloads/Nguyen2007"

def main():

    # direct from command line what function is to be run

    # if the argument g is provided
    if args.g:

        # set the file path to this file
        filepath = args.g

    # otherwise
    else:

        # set the file path to the root
        filepath = root



    '''
    need if else structure to work with the 3 data structures

    if args.f == "/Users/martinhanna/Downloads/aylien-covid-news.jsonl":

        print()

    elif args.f == "":

        print()

    else:

        print()

    '''

    # declare the dataset
    dataSet = DataSet()

    # extracting the paths
    key_paths, text_paths = extractFilePath(filepath)

    # declare new columns in the df
    dataSet.df['key_paths'] = key_paths
    dataSet.df['text_paths'] = text_paths

    # declare the path
    path = dataSet.df.text_paths[0]

    # declare the text
    text = extractDocLines(path)

    # extract content
    dataSet.df['rawText'] = dataSet.df.text_paths.apply(extractDocLines)

    # extract the keywords
    dataSet.df['keywords'] = dataSet.df.key_paths.apply(extractKeywords)

    # clean the text
    dataSet.df['vsm'] = dataSet.df.rawText.apply(cleanText)

    # apply tfidf
    dataSet.model, dataSet.matrix = applyTFIDF(dataSet.df['vsm'])

    # explore the matrix with the dataset
    dataSet.dict_index, dataSet.dict_values = exploreMatrix(dataSet.matrix, dataSet.model)

    # bolster the ngrams
    # bolsterNgrams(dataSet)
    
    # print(topCorpusTerms(dataSet.matrix, dataSet.model_terms, row_id, top_n = 10)
    
    # use a dictionary like the first few weeks to find top 10 info

    # evaluate result
    evaluateModel(dataSet)


# magic method to run the main function
if __name__ == "__main__":
    main()

# print the time of the program
print("\n" + 40*"#")
print(time.time() - start)
print(40*"#" + "\n")
