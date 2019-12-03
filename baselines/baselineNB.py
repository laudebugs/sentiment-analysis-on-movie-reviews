"""
A Naive Bayes system for performing sentiment analysis
I am using the text book to develop the system. (Speech and Language Processing, Chapter 4)

"""


from __future__ import division
import nltk
import numpy as np
import os

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
# A dictionary that will hold key value pairs with keys being a word that occurs in all the documents
#  and values being a list [0 for k in range(5)] of the number of instances the word appears as an instance of class i, such that 0<=i<k

V = {}
# A list of the occurences of words in a particular class
classcounts = [ 0 for i in range (5)]

# logprior is a list of the probabilities of all classes in the list i.e. how often is a phrase of a particular sentiment?
logprior = [ 0 for i in range (5)]
# loglikelihood is a dictionary of all the words in the vocabulary representing the probabilities that a word occurs as an instance of each of the classes
loglikelihood = {}

# Loop throughout the entire training data.
for i in range(len_train):
    # Split the tab separated values into a list so that we can isolate the data on each line.
    # The data is in the following format: [PhraseId, SentenceId, Phrase, Sentiment]
    values = training_file_lines[i].split("\t")
    phrase = values[2]
    sentiment = int(values[3])
    # Tokenize the phrase
    words = nltk.word_tokenize(phrase)
    # Analyse the phrase
    """
    Note: This analysis assumes that for each phrase that indicates a particular sentiment, all the words in the phrase will be marked as bearing that sentiment.
    One approach would be analysing the single words, then moving on up the tree. However, assuming that all the data is parsed correctly, there would be no new
    words when we move up the tree.

    """
    for w in words:
        # If the word was not previously in the Vocabulary
        if w not in V:
            #Initialize the list for word w
            V[w] = [ 0 for i in range (5)]
        # Increment the instances of a word appearing in the context of the sentiment x
        classcounts[sentiment]+=1
        # Increment the number of times word w appears as an instance of sentiment x
        V[w][sentiment]+=1

    # Done looping through training data


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
