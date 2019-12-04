"""

The evaluation algorithm takes as input two files: the output file and answer key,
where both files are tsv files with two columns the PhraseId and the Sentiment

The result will give an evaluation of the F-measure which is:
2/( (1/recall)+(1/precision) )

The output file and the answer key have to be the same length and in order - sorted in ascending order from the
"""
from __future__ import division
import sys


#Create a matrix for 5 classes for each of the five sentiments
M = [ [ 0 for i in range(5)] for j in range (5)]

output_file = open(sys.argv[1], "r")
answer_key  = open(sys.argv[2], "r")

output_lines = output_file.readlines()
answer_lines = answer_key.readlines()

if len(output_lines) != len(answer_lines):
    print "Warning: Output file and Answer Key are not the same length"

# Loop through the entire file. This assumes that the Phrase ID at index i will match for both the output file and the answer key
for i in range(len(output_lines)-1):
    out = output_lines[i].split("\t")
    ans = answer_lines[i].split("\t")

    if out[0]==ans[0]:
        i = int(out[1])
        j = int(ans[1])
        M[i][j] += 1
    else:
        print "Error: ID sequence does not match!\nCheck Output file\nEnding Evaluation"
        exit()


#Calculate The precision and recall for each of the sentiment classes.
# Where i is the the rows and j is the columns

recall = 0
precision = 0

for i in range(5):
    precision_sum = 0
    recall_sum = 0
    for j in range(5):
        precision_sum += M[i][j]
        recall_sum += M[j][i]
    precision += M[i][i]/precision_sum
    recall += M[i][i]/recall_sum

precision_ave = (precision/5)*100
recall_ave =( recall/5) *100
f_score = (2/((1/precision_ave)+(1/recall_ave)))

#Print the output
print("Precision: %.2f%%\nRecall: %.2f%%\nF-Score: %.2f%%" %(precision_ave, recall_ave, f_score))
