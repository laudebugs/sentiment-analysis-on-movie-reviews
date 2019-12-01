from __future__ import division
import nltk
import numpy as np
#import the training file
training_file = open("training.tsv", "r")
training_file_lines = training_file.readlines()

len_train = len(training_file_lines)

V = {}
Classifications = [[0 for j in range (2)] for k in range (5)]
for i in range(len_train):
    values = training_file_lines[i].split("\t")
    phrase = values[2]
    sentiment = int(values[3])
    words = nltk.word_tokenize(phrase)
    for w in words:
        if w not in V:
            V[w] = [0,0,0,0,0]
            Classifications[sentiment][1]+=1
        V[w][sentiment]+=1
    Classifications[sentiment][0]+=1

#Calculate P(w,c)
for word in V:
    for p_w_c in range(5):
        V[word][p_w_c] = 10000*((V[word][p_w_c]+1)/(Classifications[p_w_c][1]+len(V)))
#Calculate P(c)
print(len_train)

for i in range(len(Classifications)):
    Classifications[i][0] = np.log(Classifications[i][0]/len_train)
print V['excellent']

training_file.close()
