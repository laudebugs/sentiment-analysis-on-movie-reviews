"""
A Naive Bayes system for performing sentiment analysis
I am using the text book to develop the system. (Speech and Language Processing, Chapter 4)

"""


from __future__ import division
import nltk
import numpy as np
import os
import re

os.chdir("..")

#import the training file
training_file = open("dataset/training.tsv", "r")
training_file_lines = training_file.readlines()

#The length of the training file
len_train = len(training_file_lines)

"""
Train the naive bayes Model.
The features were implemented in the following order
1.  The number of occurrences of a particular word as an instance
    of a particular sentiment over all words that occur as instances of that particular sentiment.

"""

class Node:
    def __init__(self):
        self.phrase = None
        self.sentiment = None
        self.right = None
        self.left = None

def branch(list_phrases, i, k, root):
    #print list_phrases[k][4]

    # The base case where we have finished parsing the entire sentence
    # OR
    # We have reached the mentioned parsed phrases
    if (len(list_phrases[k][4])-1)-i==1 or k==(len(list_phrases)-1):
        # Assign values of the leaf node
        root.phrase = list_phrases[k][2]
        root.sentiment = list_phrases[k][3]
        i+=1;
        k+=1;
        return

        # In the case where there are phrases that are not listed in the dataset and therefore skipped
    elif list_phrases[0][4][i]!=list_phrases[k][4][0]:
        # Assign the leaf node
        root.phrase = list_phrases[k][2]
        root.sentiment = list_phrases[k][3]
        print(list_phrases[0][4])
        i = int(list_phrases[0][4].index(list_phrases[k][4][0]))+1
        k+=1
        return

    root.left = Node()
    # Go Left

    # Assign the root node
    k+=1
    branch(list_phrases, i, k, root.left)
    root.phrase = list_phrases[k][2]
    root.sentiment = list_phrases[k][3]
    root.right = Node()

    j = i+len(list_phrases[k][4])-1

    # Go Right
    branch(list_phrases, j, k, root.right)

# A dictionary that will hold key value pairs with keys being a word that occurs in all the documents
#  and values being a list [0 for k in range(5)] of the number of instances the word appears as an instance of class i, such that 0<=i<k

V = {}
# A list of the occurences of words in a particular class
classcounts = [ 0 for i in range (5)]

# logprior is a list of the probabilities of all classes in the list i.e. how often is a phrase of a particular sentiment?
logprior = [ 0 for i in range (5)]
# loglikelihood is a dictionary of all the words in the vocabulary representing the probabilities that a word occurs as an instance of each of the classes
loglikelihood = {}

parsed_sentences = []
sentence_parse = []
sentenceID = ''
# Loop throughout the entire training data.
for i in range(len_train):
    # Split the tab separated values into a list so that we can isolate the data on each line.
    # The data is in the following format: [PhraseId, SentenceId, Phrase, Sentiment]
    phrase_values = training_file_lines[i].split("\t")
    #Add a tokenized form of the phrase to the end of the phrase_values
    phrase_values.append(re.split("\s+", phrase_values[2]))

    #if this is the first phrase to be parsed
    if i==0:
        sentenceID = phrase_values[1]
        sentence_parse.append(phrase_values)
    # if the sentence ID changes or you are at the end of the file,
    # Append the parsed sentences to the list of sentences
    if phrase_values[1]!=sentenceID or i==len_train-1:
        # Append the parsed sentences to the list of sentences
        parsed_sentences.append(sentence_parse)
        # Initialize the list parsed sentence
        sentence_parse = []
        # Assign the current sentence ID to the next sentence
        sentenceID = phrase_values[1]
    # Add this phrase to the parsed sentences
    sentence_parse.append(phrase_values)



    """
    # Tokenize the phrase
    all_words = nltk.word_tokenize(phrase)
    words = []
    #Cap no of words instances in a document at 1, i.e. do not include repeated words
    for word in all_words:
        if word not in words:
            words.append(word)
    # Analyse the phrase

    Note: This analysis assumes that for each phrase that indicates a particular sentiment, all the words in the phrase will be marked as bearing that sentiment.
    An example would be: for words that are neutral, they may be marked as bearing a sentiment of negative or positive, such as
    One approach would be analysing the single words, then moving on up the tree. However, assuming that all the data is parsed correctly, there would be no new
    words when we move up the tree.




    for w in words:
        # If the word was not previously in the Vocabulary
        if w not in V:
            #Initialize the list for word w
            V[w] = [ 0 for i in range (5)]
        # Increment the instances of a word appearing in the context of the sentiment x
        classcounts[sentiment]+=1
        # Increment the number of times word w appears as an instance of sentiment x
        V[w][sentiment]+=1
    """
    # Done looping through training data

Treebank = []
for i in range(len(parsed_sentences)):
    tree = Node()
    branch(parsed_sentences[i], 0, 0, tree)
    Treebank.append(tree)

# Calculate prior probabilities P(c): logprior
for c in range(5):
    logprior[c] = np.log(classcounts[c]/len_train)

# Calculate the likelihood probabilities P(w, c): loglikelihood
for word in V:
    # Initialize the loglikelihood probabilities for each class
    loglikelihood[word] = [ 0 for i in range (5)]
    for c in range(5):
        # Calculate the likelihood values with Laplace smoothing.
        loglikelihood[word][c] = np.log( (V[word][c]+1)/(classcounts[c]+len(V)) )

# Done training


### Run the trainied program against the development file ###

#Open the development file
dev_file = open("dataset/development.tsv", "r")
dev_file_lines = dev_file.readlines()

# Initialize a variable to hold the output
dev_output = ""
# Loop through the lines and assign sentiments on the go.
for line in dev_file_lines:
    # Initialize the probabilities of all the classes for each particular phrase.
    sum = [ 0 for i in range (5)]
    # Add the logprior probabilities
    for i in range(5):
        sum[i] += logprior[i]
    # Split the data, the phrase will be contained at the index 2
    values = line.split("\t")
    phrase = values[2]
    tokens = nltk.word_tokenize(phrase)
    #Loop through the tokens
    for word in tokens:
        # If the word is in the vocabulary, give it a sentiment, otherwise, skip it
        if word in V:
            # Loop through all the classes adding probabilities.
            for c in range(5):
                sum[c] += V[word][c]

    #Add the assigned sentiment to the output file in the format: PhraseId, Sentiment
    dev_output += values[0] + "\t" + str(sum.index(max(sum))) + "\n"

# Print the output to a file.
NB_file_out = open("outputs/NB_file_output.tsv", "w+")
NB_file_out.write(dev_output)


# Close the files
training_file.close()
dev_file.close()
NB_file_out.close()
