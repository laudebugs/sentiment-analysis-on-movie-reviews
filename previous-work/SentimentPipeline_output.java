import edu.stanford.nlp.ling.*;
import edu.stanford.nlp.neural.rnn.RNNCoreAnnotations;
import edu.stanford.nlp.pipeline.*;
import edu.stanford.nlp.sentiment.SentimentCoreAnnotations;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.util.CoreMap;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.time.Duration;
import java.time.Instant;
import java.util.*;

/* From Linux server inside the test/CoreNLP/bin folder */
// java -cp ../../CoreNLP/bin:../lib SentimentPipeline 3

/* From local machine inside the CoreNLP folder */
// scp input1.txt sc6220@access.cims.nyu.edu:test/CoreNLP/bin

/*

How To Update Class file on crunchy:

1. From local machines, scp SentimentPipeline.java sc6220@access.cims.nyu.edu:/test/CoreNLP/bin
2. Connect to crunchy and cd to test/CoreNLP/bin
3. From crunchy, javac -cp "../lib" SentimentPipeline.java

 */

public class SentimentPipeline_output {
    public static void main(String[] args) throws IOException {
        //String text = new String(Files.readAllBytes(Paths.get("input.txt"));
        args[0] = "30";
//        args[0] = "2";
        int fnum = Integer.parseInt(args[0]);
        Instant start = Instant.now();

        int[] fileNums = new int[fnum];
        for (int i = 0; i < fnum; i++) fileNums[i] = (i + 1);

        double[] scoreOfLongestSentence = new double[fnum];
        double[] realScores = new double[fnum];
        int[][] scores = new int[fnum][5];

        getScores(fnum, realScores, scores, scoreOfLongestSentence);

        // 5 category classifications
        fiveClassifications(scores, realScores, scoreOfLongestSentence, fileNums);

        // 3 category classifications
        threeClassifications(scores, realScores, scoreOfLongestSentence, fileNums);

        // 2 binary classifications
        twoClassifications(scores, realScores, scoreOfLongestSentence, fileNums);

        long timeElapsed = Duration.between(start, Instant.now()).toMillis();  //in millis
        System.out.println("Time taken: " + timeElapsed + "ms (" + (timeElapsed / 1000) + "s)");
    }

    /*
    Methods for Classification with Two classes (0 if stars is < 2.5, 1 if stars are >= 2.5)
     */

    private static void twoClassifications(int[][] scores, double[] realScores, double[] scoreOfLongestSentence, int[] fileNums) {
        // Set scores to 0 (schlect) or 1 (okay)
        double[] rp = new double[realScores.length];
        for (int i = 0; i < rp.length; i++) {
            double score = realScores[i];
            if (score < 2.5) {
                score = 0;
            } else if (score >= 2.5) {
                score = 1;
            }
            rp[i] = score;
        }

        double[] lp = new double[scoreOfLongestSentence.length];
        for (int i = 0; i < lp.length; i++) {
            double score = scoreOfLongestSentence[i];
            if (score < 3) {
                score = 0;
            } else if (score >= 3) {
                score = 1;
            }

            lp[i] = score;
        }

        int[][] sp = new int[scores.length][2];
        for (int i = 0; i < scores.length; i++) {
            for (int j = 0; j < scores[i].length; j++) {
                if (j == 0 || j == 1) sp[i][0] += scores[i][j];
                if (j == 2 || j == 3 || j == 4) sp[i][1] += scores[i][j];
            }
        }


        twoAverageScores(sp, rp, fileNums);
        // Since we are dealing with binary it would make sense to round the results, especially since our realscores is rounded to 0 or 1 we won't ever have a decimal
        twoAverageScoresRoundedPrediction(sp, rp, fileNums);
        twoMaxScores(sp, rp, fileNums);
        twoLongestSentenceScore(lp, rp, fileNums);
    }

    private static void twoAverageScores(int[][] scores, double[] realScores, int[] fileNums) {
        double[] predScores = new double[realScores.length];

        for (int f = 0; f < fileNums.length; f++) {
            int totalScore = 0;
            double numScores = 0;

            int[] scoreArr = scores[f];
            for (int i = 0; i < scoreArr.length; i++) {
                numScores += scoreArr[i];
                totalScore += i * scoreArr[i];
            }

            predScores[f] = totalScore / numScores;
        }

        printSummary(realScores, predScores, fileNums, "Average Sentiment Score for Binary Classification");
    }

    private static void twoAverageScoresRoundedPrediction(int[][] scores, double[] realScores, int[] fileNums) {
        double[] predScores = new double[realScores.length];

        for (int f = 0; f < fileNums.length; f++) {
            int totalScore = 0;
            double numScores = 0;

            int[] scoreArr = scores[f];
            for (int i = 0; i < scoreArr.length; i++) {
                numScores += scoreArr[i];
                totalScore += i * scoreArr[i];
            }

            predScores[f] = Math.round(totalScore / numScores);
        }

        printSummary(realScores, predScores, fileNums, "Average Sentiment Score for Binary Classification with Rounded Predicted Scores");
    }

    private static void twoMaxScores(int[][] scores, double[] realScores, int[] fileNums) {
        double[] predScores = new double[realScores.length];

        for (int f = 0; f < fileNums.length; f++) {
            int maxNums = 0;
            int maxSentiment = 0;

            int[] scoreArr = scores[f];
            for (int i = 0; i < scoreArr.length; i++) {
                // In the case of ties, we favor the higher rated one
                if (scoreArr[i] >= maxNums) {
                    maxSentiment = i;
                }
            }

            predScores[f] = maxSentiment;
        }

        printSummary(realScores, predScores, fileNums, "Most Common Sentiment Score for Binary Classification");
    }

    private static void twoLongestSentenceScore(double[] scoreOfLongestSentence, double[] realScores, int[] fileNums) {
        printSummary(realScores, scoreOfLongestSentence, fileNums, "Sentiment of the Longest Sentence for Binary Classification");
    }

    /*
    End of Binary classification
     */

    /*
    Methods for Classification with Three classes (removing extremes, only 2, 3, or 4 sentiment)
     */

    private static void threeClassifications(int[][] scores, double[] realScores, double[] scoreOfLongestSentence, int[] fileNums) {
        // Combine the extreme scores, to get 2, 3, or 4, where 2 is stars 1 and 2, 3 is 3, and 4 is 4 and 5

        // Round the extremes instead of just shifting them by 1
        double[] rp = new double[realScores.length];
        for (int i = 0; i < rp.length; i++) {
            double score = realScores[i];
            if (score > 4) {
                score = 4;
            } else if (score < 2) {
                score = 2;
            }
            rp[i] = score;
        }

        double[] roundedScores = new double[rp.length];
        for (int i = 0; i < rp.length; i++) {
            roundedScores[i] = Math.round(rp[i]);
        }

        double[] lp = new double[scoreOfLongestSentence.length];
        for (int i = 0; i < lp.length; i++) {
            double score = scoreOfLongestSentence[i];
            if (score == 1) {
                score = 2;
            } else if (score == 5) {
                score = 4;
            }

            lp[i] = score;
        }

        int[][] sp = new int[scores.length][3];
        for (int i = 0; i < scores.length; i++) {
            for (int j = 0; j < scores[i].length; j++) {
                if (j == 0 || j == 1) sp[i][0] += scores[i][j];
                if (j == 2) sp[i][1] += scores[i][j];
                if (j == 3 || j == 4) sp[i][2] += scores[i][j];
            }
        }

        threeAverageScoresCombineExtremes(sp, rp, fileNums);
        threeMaxScoresCombineExtremes(sp, rp, fileNums);
        threeMaxScoresCombineExtremesRoundedScores(sp, roundedScores, fileNums);
        threeLongestSentenceScoreCombineExtremes(lp, rp, fileNums);
        threeLongestSentenceScoreCombineExtremesRoundedScores(lp, roundedScores, fileNums);
    }

    private static void threeAverageScoresCombineExtremes(int[][] scores, double[] realScores, int[] fileNums) {
        double[] predScores = new double[realScores.length];

        for (int f = 0; f < fileNums.length; f++) {
            int totalScore = 0;
            double numScores = 0;

            int[] scoreArr = scores[f];
            for (int i = 0; i < scoreArr.length; i++) {
                numScores += scoreArr[i];
                totalScore += (i + 2) * scoreArr[i];
            }

            predScores[f] = totalScore / numScores;
        }

        printSummary(realScores, predScores, fileNums, "Average Sentiment Score for Three Classes with Combined Extremes");
    }

    private static void threeMaxScoresCombineExtremes(int[][] scores, double[] realScores, int[] fileNums) {
        double[] predScores = new double[realScores.length];

        for (int f = 0; f < fileNums.length; f++) {
            int maxNums = 0;
            int maxSentiment = 0;

            int[] scoreArr = scores[f];
            for (int i = 0; i < scoreArr.length; i++) {
                // In the case of ties, we favor the higher rated one
                if (scoreArr[i] >= maxNums) {
                    maxSentiment = (i + 2);
                }
            }

            predScores[f] = maxSentiment;
        }

        printSummary(realScores, predScores, fileNums, "Most Common Sentiment Score for Three Classifiers with Combined Extremes");
    }

    private static void threeMaxScoresCombineExtremesRoundedScores(int[][] scores, double[] realScores, int[] fileNums) {
        double[] predScores = new double[realScores.length];

        for (int f = 0; f < fileNums.length; f++) {
            int maxNums = 0;
            int maxSentiment = 0;

            int[] scoreArr = scores[f];
            for (int i = 0; i < scoreArr.length; i++) {
                // In the case of ties, we favor the higher rated one
                if (scoreArr[i] >= maxNums) {
                    maxSentiment = (i + 2);
                }
            }

            predScores[f] = maxSentiment;
        }

        printSummary(realScores, predScores, fileNums, "Most Common Sentiment Score for Three Classifiers with Combined Extremes and Rounded Scores");
    }

    private static void threeLongestSentenceScoreCombineExtremes(double[] scoreOfLongestSentence, double[] realScores, int[] fileNums) {
        printSummary(realScores, scoreOfLongestSentence, fileNums, "Sentiment of the Longest Sentence for Three Classifications with Combined Extremes");
    }

    private static void threeLongestSentenceScoreCombineExtremesRoundedScores(double[] scoreOfLongestSentence, double[] realScores, int[] fileNums) {
        printSummary(realScores, scoreOfLongestSentence, fileNums, "Sentiment of the Longest Sentence for Three Classifications with Combined Extremes and Rounded Scores");
    }

    /*
    End of Three classification
     */

    /*
    Methods for Classification with Five classes (the normal star scale)
     */

    public static void fiveClassifications(int[][] scores, double[] realScores, double[] scoreOfLongestSentence, int[] fileNums) {
        double[] roundedScores = new double[realScores.length];
        for (int i = 0; i < realScores.length; i++) {
            roundedScores[i] = Math.round(realScores[i]);
        }

        fiveAverageScores(scores, realScores, fileNums);
        fiveMaxScores(scores, realScores, fileNums);
        fiveMaxScoresRoundedScore(scores, roundedScores, fileNums);
        fiveLongestSentenceScore(scoreOfLongestSentence, realScores, fileNums);
        fiveLongestSentenceScoreRoundedScore(scoreOfLongestSentence, roundedScores, fileNums);
    }

    public static void fiveAverageScores(int[][] scores, double[] realScores, int[] fileNums) {
        double[] predScores = new double[realScores.length];

        for (int f = 0; f < fileNums.length; f++) {
            int totalScore = 0;
            double numScores = 0;

            int[] scoreArr = scores[f];
            for (int i = 0; i < scoreArr.length; i++) {
                numScores += scoreArr[i];
                totalScore += (i + 1) * scoreArr[i];
            }

            predScores[f] = totalScore / numScores;
        }
//        Sort
//        sortResultsBy(realScores, predScores, fileNums, 5);
        printSummary(realScores, predScores, fileNums, "Average Sentiment Score for Five Stars");
//        Sort back to default
//        sortResultsBy(realScores, predScores, fileNums, 0);
    }

    public static void fiveMaxScores(int[][] scores, double[] realScores, int[] fileNums) {
        double[] predScores = new double[realScores.length];

        for (int f = 0; f < fileNums.length; f++) {
            int maxNums = 0;
            int maxSentiment = 0;

            int[] scoreArr = scores[f];
            for (int i = 0; i < scoreArr.length; i++) {
                // In the case of ties, we favor the higher rated one
                if (scoreArr[i] >= maxNums) {
                    maxSentiment = (i + 1);
                }
            }

            predScores[f] = maxSentiment;
        }

        printSummary(realScores, predScores, fileNums, "Most Common Sentiment Score for Five Stars");
    }

    public static void fiveMaxScoresRoundedScore(int[][] scores, double[] realScores, int[] fileNums) {
        double[] predScores = new double[realScores.length];

        for (int f = 0; f < fileNums.length; f++) {
            int maxNums = 0;
            int maxSentiment = 0;

            int[] scoreArr = scores[f];
            for (int i = 0; i < scoreArr.length; i++) {
                // In the case of ties, we favor the higher rated one
                if (scoreArr[i] >= maxNums) {
                    maxSentiment = (i + 1);
                }
            }

            predScores[f] = maxSentiment;
        }

        printSummary(realScores, predScores, fileNums, "Most Common Sentiment Score for Five Stars with Rounded Scores");
    }

    public static void fiveLongestSentenceScore(double[] scoreOfLongestSentence, double[] realScores, int[] fileNums) {
        printSummary(realScores, scoreOfLongestSentence, fileNums, "Sentiment of the Longest Sentence for Five Stars");
    }

    public static void fiveLongestSentenceScoreRoundedScore(double[] scoreOfLongestSentence, double[] realScores, int[] fileNums) {
        printSummary(realScores, scoreOfLongestSentence, fileNums, "Sentiment of the Longest Sentence for Five Stars with Rounded Scores");
    }

    /*
    End of Five classification
     */

    public static void getScores(int fnum, double[] realScores, int[][] scores, double[] scoreOfLongestSentence) throws FileNotFoundException {
        for (int f = 0; f < fnum; f++) {
            int num = f + 1;
//            num = 16;
            File file = new File("input" + num + ".txt");
            System.out.println("File " + (f + 1));
            Scanner sc = new Scanner(file);
            String text = "";
            double score = 3;       //default neutral, but of course this will never be used if input is correct
            if (sc.hasNextDouble()) {
                score = sc.nextDouble();     //the system takes on an actual rating of the rotton tomato set
                if (score < 1) score = 1;
            }
            while (sc.hasNextLine()) {
                text += sc.nextLine() + " ";
            }
            text = text.replaceAll("[ \r\n]+", " ");

            // set up pipeline properties
            Properties props = new Properties();
            props.setProperty("annotators", "tokenize, ssplit, pos, lemma, ner, parse, dcoref, sentiment");
            // set the list of annotators to run
//        //props.setProperty("annotators", "tokenize,ssplit,pos,lemma,ner,parse,depparse,coref,kbp,quote");
            // set a property for an annotator, in this case the coref annotator is being set to use the neural algorithm
            props.setProperty("coref.algorithm", "neural");
            // build pipeline
            StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
            // create a document object
            CoreDocument document = new CoreDocument(text);
            // annnotate the document
            pipeline.annotate(document);
            int sentnum = document.sentences().size();

            int maxSentenceLength = 0;

            for (int i = 0; i < document.sentences().size(); i++) {
                System.out.println("Sentence " + (i + 1) + ": " + document.sentences().get(i));
                CoreSentence sentence = document.sentences().get(i);
//            if(unrelated(sentence)){
//                System.out.println("Unrelated sentence omitted.........");
//               continue;
//            }
                Annotation annotation = new Annotation(sentence.text());
                pipeline.annotate(annotation);

                List<CoreMap> sentences = annotation.get(CoreAnnotations.SentencesAnnotation.class);

                CoreMap sentMap = sentences.get(0);
                String sent = sentMap.get(SentimentCoreAnnotations.SentimentClass.class);
                System.out.println("Sentiment: " + sent);
                double sentScore;
                if (sent.equals("Very positive")) {
                    sentScore = 5;
                    scores[f][4]++;
                } else if (sent.equals("Positive")) {
                    sentScore = 4;
                    scores[f][3]++;
                } else if (sent.equals("Neutral")) {
                    sentScore = 3;
                    scores[f][2]++;
                } else if (sent.equals("Negative")) {
                    sentScore = 2;
                    scores[f][1]++;
                } else if (sent.equals("Very negative")) {
                    sentScore = 1;
                    scores[f][0]++;
                } else {
                    System.out.println("Unknown sentiment " + sent);
                    sentScore = -1;   //sentinemnt unclear
                }

                if (sentence.text().length() > maxSentenceLength) {
                    maxSentenceLength = sentence.text().length();
                    scoreOfLongestSentence[f] = sentScore;
                }
//            if(isSummary(sentence)){                //giving summarial sentence much more weight if they exist, note: this function will grow exponentially towards the end but let's deal with that later
//                System.out.println(sentence.text()+" is summary "+sentScore);
//                avgSent+=sentScore*document.sentences().size();
//                sentnum+=document.sentences().size();
//            }
//            else {
//                avgSent += sentScore;
//            }
                //implement sentiment analysis in this, and then check POS (figure out POS within the sentence)
            }

            realScores[f] = score;
        }
    }

    public static void sortResultsBy(double[] realScores, double[] predScores, int[] fileNums, int sortNum) {
        // Bubble sort cuz I like buBBles

        boolean done = false;
        while (!done) {
            done = true;
            for (int i = 0; i < fileNums.length - 1; i++) {
                // Default sort ascending file numbers
                boolean condition = fileNums[i] > fileNums[i + 1];
                switch (sortNum) {
                    case 2:
                        // Sort ascending real scores
                        condition = realScores[i] > realScores[i + 1];
                        break;
                    case 3:
                        // Sort ascending predicted scores
                        condition = predScores[i] > predScores[i + 1];
                        break;
                    case 4:
                        // Sort descending percent difference
                        condition = Math.abs((predScores[i] - realScores[i]) / realScores[i]) < Math.abs((predScores[i + 1] - realScores[i + 1]) / realScores[i + 1]);
                        break;
                    case 5:
                        // Sort descending numerical difference
                        condition = Math.abs((predScores[i] - realScores[i])) < Math.abs((predScores[i + 1] - realScores[i + 1]));
                        break;
                }

                if (condition) {
                    double temp1 = realScores[i];
                    realScores[i] = realScores[i + 1];
                    realScores[i + 1] = temp1;
                    double temp2 = predScores[i];
                    predScores[i] = predScores[i + 1];
                    predScores[i + 1] = temp2;
                    int temp3 = fileNums[i];
                    fileNums[i] = fileNums[i + 1];
                    fileNums[i + 1] = temp3;
                    done = false;
                }
            }
        }
    }

    public static void printSummary(double[] realScores, double[] predScores, int[] fileNums, String title) {
        int fnum = fileNums.length;

        String header = "===== Summary for " + title + " =====";
        System.out.println(header);
        System.out.print("File Nums:\t\t\t");
        for (int f = 0; f < fnum; f++) {
            String print = " " + fileNums[f];
            System.out.print(print);
            if (f < fnum - 1) {
                // Black magic formatting
                System.out.print(",");
                for (int i = print.length(); i < 8; i++) System.out.print(" ");
                System.out.print("\t");
            }
        }
        System.out.println();

        System.out.print("Real Score:\t\t\t");
        for (int i = 0; i < realScores.length; i++) {
            System.out.print(dec3(realScores[i]));
            if (i < realScores.length - 1) System.out.print(",   \t");
        }
        System.out.println();

        System.out.print("Predicted Score:\t");
        for (int i = 0; i < predScores.length; i++) {
            System.out.print(dec3(predScores[i]));
            if (i < realScores.length - 1) System.out.print(",   \t");
        }
        System.out.println();
        System.out.println();

        double totalP = 0;
        double totalPAbs = 0;
        System.out.print("Difference (%):\t\t");
        for (int f = 0; f < fnum; f++) {
            double percent = (predScores[f] - realScores[f]) / realScores[f];
            if (realScores[f] == 0.0) {
                if ((predScores[f] - realScores[f]) == 0) {
                    percent = 0;
                } else {
                    percent = 1;
                }
            }
            String print = dec3(100 * percent) + "%";
            totalP += percent;
            totalPAbs += Math.abs(percent);
            System.out.print(print);
            if (f < fnum - 1) {
                // Black magic formatting
                System.out.print(",");
                for (int i = print.length(); i < 8; i++) System.out.print(" ");
                System.out.print("\t");
            }
        }
        System.out.println();

        double totalD = 0;
        double largestGreater = -1;
        double greaterActual = 0;
        double greaterPredicted = 0;
        int greaterFile = 0;

        double largestLesser = 1;
        double lesserActual = 0;
        double lesserPredicted = 0;
        int lesserFile = 0;

        double totalDAbs = 0;
        System.out.print("Difference (#):\t\t");
        for (int f = 0; f < fnum; f++) {
            double numError = (predScores[f] - realScores[f]);
            if (numError > largestGreater) {
                largestGreater = numError;
                greaterActual = realScores[f];
                greaterPredicted = predScores[f];
                greaterFile = fileNums[f];
            }
            if (numError < largestLesser) {
                largestLesser = numError;
                lesserActual = realScores[f];
                lesserPredicted = predScores[f];
                lesserFile = fileNums[f];
            }
            totalD += predScores[f] - realScores[f];
            totalDAbs += Math.abs(predScores[f] - realScores[f]);
            System.out.print(dec3(numError));
            if (f < fnum - 1) System.out.print(",   \t");
        }
        System.out.println();
        System.out.println();

        System.out.println("Avg Percent Difference:\t\t\t\t\t" + dec3(100 * totalP / (double) fnum) + "%");
        System.out.println("Avg Percent Difference (abs):\t\t\t" + dec3(100 * totalPAbs / (double) fnum) + "%");

        // Abs is the more useful one because it's the avg of every error regardless of sign
        System.out.println("Avg Numerical Difference (abs):\t\t" + dec3(totalDAbs / (double) fnum));

        // This error shows you if you're on average more predicting things lower or higher than they should be (signs will fighT each other out)
        // *** Closer to 0 means that we are overestimating as much as we are underestimating
        System.out.println("Avg Numerical Difference:\t\t\t" + dec3(totalD / (double) fnum));
        System.out.println("Largest Overestimated Difference:\t" + dec3(largestGreater) + " ( Actual: " + dec3(greaterActual) + " Predicted: " + dec3(greaterPredicted) + " in File: " + greaterFile + " )");
        System.out.println("Largest Underestimated Difference:\t" + dec3(largestLesser) + " ( Actual: " + dec3(lesserActual) + " Predicted: " + dec3(lesserPredicted) + " in File: " + lesserFile + " )");
        StringBuilder footer = new StringBuilder();
        while (footer.length() < header.length()) footer.append("=");
        System.out.println(footer);
        System.out.println();
    }

    public static String dec3(Number n) {
        NumberFormat dec3 = new DecimalFormat("0.000");
        return (n.toString().startsWith("-")) ? dec3.format(n) : " " + dec3.format(n);
    }

    public static boolean isSummary(CoreSentence sentence) {
        if (sentence.text().contains("overall") || sentence.text().contains("all in all")) {
            return true;
        }
        return false;
    }


    public static boolean unrelated(CoreSentence sentence) {

        //TODO: check subject of sentence
        if (sentence.text().matches("in (this|the)(\\s(a-z)*\\s)(movie||film)")) {
            return true;
        } else if (sentence.text().contains("sequal")) {
            return true;
        } else if (sentence.text().contains("review")) {        //eliminate comparison to other reviews
            return true;
        }
        for (int j = 0; j < sentence.tokens().size(); j++) {
            CoreLabel token = sentence.tokens().get(j);
            //in the case that the subject of sentence is the film, check if it is describing plot (verb, e.g.: shows, presentes, etc.) or evaluating the film (adj.: e.g.: is ...,
            if (token.word().equals("movie") || token.word().equals("film")) {
                System.out.println(sentence.tokens().get(j + 1) + " tag: " + sentence.posTags().get(j + 1));

                if (sentence.posTags().get(j + 1).equals("JJ") || sentence.posTags().get(j + 1).equals("JJR") || sentence.posTags().get(j + 1).equals("JJS")) {
                    return false;
                } else if (sentence.tokens().get(j + 1).word().equals("is")) {
                    return false;
                } else if (sentence.posTags().get(j + 1).equals("VB") || sentence.posTags().get(j + 1).equals("VBP") || sentence.posTags().get(j + 1).equals("VBZ") || sentence.posTags().get(j + 1).equals("VBD") || sentence.posTags().get(j + 1).equals("VBG") || sentence.posTags().get(j + 1).equals("VBN")) {
                    return true;
                } else {
                    return false;
                }
            }
        }
        return false;
    }

    public static String subject(CoreSentence sentence) {
        return sentence.sentiment();
    }
}
