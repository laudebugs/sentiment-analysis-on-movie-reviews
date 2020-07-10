import edu.stanford.nlp.coref.data.CorefChain;
import edu.stanford.nlp.ling.*;
import edu.stanford.nlp.ie.util.*;
import edu.stanford.nlp.pipeline.*;
import edu.stanford.nlp.semgraph.*;
import edu.stanford.nlp.sentiment.SentimentCoreAnnotations;
import edu.stanford.nlp.simple.Token;
import edu.stanford.nlp.trees.*;
import edu.stanford.nlp.util.CoreMap;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

/* From Linux server inside the test/CoreNLP/bin folder */
// java -cp ../../CoreNLP/bin:../lib SentimentPipeline 3

/* From local machine inside the CoreNLP folder (where the input files are) */
// scp input1.txt sc6220@ac cess.cims.nyu.edu:test/CoreNLP/bin

/*

How To Update Class file on crunchy:

1. From local machines, scp SentimentPipeline.java sc6220@access.cims.nyu.edu:/test/CoreNLP/bin
2. Connect to crunchy and cd to test/CoreNLP/bin
3. From crunchy, javac -cp "../lib" SentimentPipeline.java

 */

public class SentimentPipeline {
    public static void main(String[] args) throws IOException {
        //String text = new String(Files.readAllBytes(Paths.get("input.txt"));
        int fnum = Integer.parseInt(args[0]);
        double[] predScores = new double[fnum];
        double[] realScores = new double[fnum];
        ;
        for(int f = 0; f < fnum; f++){
            File file = new File("input"+(f+1)+".txt");
            System.out.println("File "+(fnum+1));
            Scanner sc = new Scanner(file);
            String text = "";
            Double score = 3.0;       //default neutral, but of course this will never be used if input is correct
            if(sc.hasNextDouble()){
                score = sc.nextDouble();     //the system takes on an actual rating of the rotton tomato set
            }
            while(sc.hasNextLine()){
                text+=sc.nextLine().toLowerCase()+" ";
            }

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
            double avgSent = 0;
            int sentnum = document.sentences().size();

            for(int i = 0; i < document.sentences().size(); i++){

                System.out.println("Sentence "+(i+1)+": "+document.sentences().get(i));

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
                System.out.println("Sentiment: "+sent);
                double sentScore;
                if (sent.equals("Very positive")) {
                    sentScore = 5;
                } else if (sent.equals("Positive")) {
                    sentScore = 4;
                } else if (sent.equals("Neutral")) {
                    sentScore = 3;

                } else if (sent.equals("Negative")) {
                    sentScore =2;
                } else if (sent.equals("Very negative")) {
                    sentScore = 1;
                } else {
                    System.out.println("Unknown sentiment " + sent);
                    sentScore = -1;   //sentinemnt unclear
                }
//            if(isSummary(sentence)){                //giving summarial sentence much more weight if they exist, note: this function will grow exponentially towards the end but let's deal with that later
//                System.out.println(sentence.text()+" is summary "+sentScore);
//                avgSent+=sentScore*document.sentences().size();
//                sentnum+=document.sentences().size();
//            }
//            else {
//                avgSent += sentScore;
//            }
                avgSent += sentScore;

                //implement sentiment analysis in this, and then check POS (figure out POS within the sentence)
            }

            avgSent/=sentnum;
            avgSent += (avgSent-2.76)*0.3091;
            realScores[f] = score;
            predScores[f] = avgSent;

        }
        System.out.println("Summary: ");
        System.out.print("Real Score: \t");
        for(double sc : realScores){
            System.out.print(","+sc);
        }
        System.out.println();
        System.out.print("Predicted Score: ");
        for(double sc : predScores){
            System.out.print(","+sc);
        }
        System.out.println();
        System.out.print("Difference: \t");
        for(int f = 0; f<fnum; f++){
            System.out.print(","+(predScores[f]-realScores[f])/realScores[f]);
        }



    }


    public static boolean isSummary(CoreSentence sentence){
        if(sentence.text().contains("overall")||sentence.text().contains("all in all")){
            return true;
        }
        return false;
    }


    public static boolean unrelated(CoreSentence sentence){

        //TODO: check subject of sentence
        if(sentence.text().matches("in (this|the)(\\s(a-z)*\\s)(movie||film)")){
            return true;
        }

        else if(sentence.text().contains("sequal")){
            return true;
        }
        else if(sentence.text().contains("review")){        //eliminate comparison to other reviews
            return true;
        }
        for(int j = 0; j < sentence.tokens().size(); j++){
            CoreLabel token = sentence.tokens().get(j);
            //in the case that the subject of sentence is the film, check if it is describing plot (verb, e.g.: shows, presentes, etc.) or evaluating the film (adj.: e.g.: is ...,
            if(token.word().equals("movie")||token.word().equals("film")){
                System.out.println(sentence.tokens().get(j+1)+" tag: "+sentence.posTags().get(j+1));

                if(sentence.posTags().get(j+1).equals("JJ")||sentence.posTags().get(j+1).equals("JJR")||sentence.posTags().get(j+1).equals("JJS")){
                    return false;
                }
                else if(sentence.tokens().get(j+1).word().equals("is")){
                    return false;
                } else if(sentence.posTags().get(j+1).equals("VB")||sentence.posTags().get(j+1).equals("VBP")||sentence.posTags().get(j+1).equals("VBZ")||sentence.posTags().get(j+1).equals("VBD")||sentence.posTags().get(j+1).equals("VBG")||sentence.posTags().get(j+1).equals("VBN")){
                    return true;
                }
                else{
                    return false;
                }
            }
        }
        return false;
    }


}
