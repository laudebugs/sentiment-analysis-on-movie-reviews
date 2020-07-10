# 🗄️The Data

The original files obtained from [Kaggle](https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/data) contains the following files:

1. sampleSubmission.csv
2. test.tsv
3. train.tsv

## Creating a development set

In order to tune parameters/features as we develop the algorithm, it was necessary to create a development set from [the original training set](original/train.tsv). We used [a simple script](..\scripts\split_test_set.py) to split the test.tsv file into training.tsv and development.tsv with a ratio of 3:1 respectively of the whole original dataset The initial split of the train.tsv is 75% training.tsv and 25% development.tsv. This ratio can be adjusted in the split_test_set.py to different values.

### Folder Structure

- training.tsv - the training set
- development.tsv - the development set
- devanskey.tsv - the answer key to the development set
- test.tsv - the test set to run the final algorithm against before submission
