import numpy as np
import string
import nltk
import matplotlib.pyplot as plt

training_file = open("dataset/train.tsv", "r")
training_file_lines = training_file.readlines()

len_train = len(training_file_lines)

labels = ['negative', 'somewhat\nnegative','neutral','somewhat\npositive','positive']

sentiment_lengths = [[]for sl in range(5)]
#Obtain the phrase from each new sentence, where the sentence changes
# The values contain the following fields: PhraseId, SentenceId, Phrase,	Sentiment
list_sentences = [[]for ls in range(5)]
sentenceID = 0
sentence = ""
for i in range(1, len_train):
    values = training_file_lines[i].split('\t')

    if int(values[1])>sentenceID:
        sentence = values[2]
        sentiment = int(values[3])
        list_sentences[sentiment].append(sentence)
        #Remove punctuation from sentence
        sentence = sentence.translate(str.maketrans('', '', string.punctuation))
        #Add sentence length
        slength = len(nltk.word_tokenize(sentence))
        #Add sentiment
        sentiment_lengths[sentiment].append(slength)
    sentenceID = int(values[1])
sentence_file = open('sentences.txt', 'w+')
for s in range(5):
    sentence_file.write(labels[s]+'\n')
    for line in list_sentences[s]:
        sentence_file.write(line+'\n')
    sentence_file.write('\n')
sentence_file.close()

"""
#Plotting the average sentence length vs sentiment
for s in range(5):
    ave_lens = 0
    for l in sentiment_lengths[s]:
        ave_lens += sentiment_lengths[s][l]
    ave_lens/=len(sentiment_lengths[s])
    plt.bar(s+1, ave_lens)
plt.ylabel("Average sentence lengths")
plt.xlabel("sentiment")
plt.show()

#Plot count of sentences for each sentiment
sentence_count = []
total_sentence_count = 0
for s in range(5):
    sentences_for_sentiment = len(sentiment_lengths[s])
    sentence_count.append(sentences_for_sentiment)
    total_sentence_count+= sentences_for_sentiment
print ("Total Sentences: " + str(total_sentence_count))
labels = ['negative'+'\n'+str(sentence_count[0]),
            'somewhat\nnegative'+'\n'+str(sentence_count[1]),
                'neutral'+'\n'+str(sentence_count[2]),
                    'somewhat\npositive'+'\n'+str(sentence_count[3]),
                        'positive'+'\n'+str(sentence_count[4])]

sizes = sentence_count
explode = (0, 0.1, 0, 0.1, 0)

fig, ax = plt.subplots()
ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=50)
ax.axis('equal')
plt.show()
"""

"Question: For how much of the training set is sentiment determined by punctuation?"
previous_phrase = ''
count=0
for n in range(1,len_train):
    if previous_phrase == '':
        previous_phrase = training_file_lines[n]
        continue
    else:
        phrase_values = training_file_lines[n].split('\t')
        curr_phrase = phrase_values[2].translate(str.maketrans('', '', string.punctuation)).strip()
        phrase_sentiment = int(phrase_values[3])

        prev_phrase_values = previous_phrase.split('\t')
        prev_phrase_sentiment = int(prev_phrase_values[3])
        prev_phrase = prev_phrase_values[2].translate(str.maketrans('', '', string.punctuation)).strip()
        if (curr_phrase==prev_phrase):
            count+=1
        previous_phrase = training_file_lines[n]
print ("Count is: " + str(count) + " out of "+ str(len_train) + " which is % .2f%% of of all the documents" %((count/len_train)*100))
