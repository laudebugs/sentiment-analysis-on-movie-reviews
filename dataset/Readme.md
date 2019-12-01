The original files obtained from the Kaggle website (https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/data) contains the following files:
1. sampleSubmission.csv
2. test.tsv
3. train.tsv

In order to create a development file to tune parameters/features as we develop algorithms,
the program split_test_set.py can be used to split the test.tsv file into training.tsv and development.tsv. The initial split of the train.tsv is 75% training.tsv and 25% development.tsv. This ratio can be adjusted in the split_test_set.py to different values.

Upon running split_test_set.py a new file, devanskey.tsv will be created that will contain the answer key to evaluate an algorithm against the development file.
