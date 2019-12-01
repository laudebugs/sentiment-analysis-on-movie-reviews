from __future__ import division
#import the training file
training_file = open("train.tsv", "r")
training_file_lines = training_file.readlines()

#Create a development file from te training file
development_file = open("development.tsv", "w+")
development_anskey = open("devanskey.tsv", "w+")
new_training_file = open("training.tsv", "w+")
len_train = len(training_file_lines)

for i in range(1, len_train):
    if (i/len_train)<0.75:
        new_training_file.write(training_file_lines[i])
    else:
        values = training_file_lines[i].split('\t')
        development_file.write(values[0]+"\t"+values[1]+"\t"+values[2]+"\n")
        development_anskey.write(values[0]+"\t"+values[3]+"\n")

training_file.close()
development_file.close()
new_training_file.close()
