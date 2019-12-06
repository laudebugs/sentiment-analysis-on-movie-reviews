In order to classify sentences on a five point scale following a challenge on Kaggle: https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews, 
we are developing a sentiment analysis classifier to classify the sentiment in the text for each phrases. 

The current Naive Bayes classifier, found in the baseline directory achieves a simple classification based on whether or not a words with a particular sentiment appears in the sentence

To run the program, simply call the program: python baselineNB.py
To evaluate, based on the evaluation algorithm we wrote, run, from the folder: evaluation: python evaluate_f_score.py test_set answer_key
          for example, to run against the development set: python evaluate_f_score.py ../dataset/development.tsv ../dataset/devanskey.tsv
