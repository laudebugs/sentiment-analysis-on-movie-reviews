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

public class SentimentPipeline {
    public static void main(String[] args) throws IOException {
        //String text = new String(Files.readAllBytes(Paths.get("input.txt"));
        File file = new File("input.txt");
        Scanner sc = new Scanner(file);
        String text = "";
        Double score;
//        if(sc.hasNextDouble()){
//            score = sc.nextDouble();     //the system takes on an actual rating of the rotton tomato set
//        }
//        else{
//            score = Double.parseDouble(sc.nextLine());
//        }
        while(sc.hasNextLine()){
            text+=sc.nextLine();
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

        for(int i = 0; i < document.sentences().size(); i++){

            System.out.println("Sentence "+(i+1)+": "+document.sentences().get(i));

            CoreSentence sentence = document.sentences().get(i);
            if(unrelated(sentence)){
                System.out.println("Unrelated sentence omitted.........");
                continue;
            }
            Annotation annotation = new Annotation(sentence.text());
            pipeline.annotate(annotation);

            List<CoreMap> sentences = annotation.get(CoreAnnotations.SentencesAnnotation.class);
            CoreMap sentMap = sentences.get(0);
            String sent = sentMap.get(SentimentCoreAnnotations.SentimentClass.class);
            System.out.println("Sentiment: "+sent);
            double sentScore;
            if (sent.equals("Very positive")) {
                sentScore = 1;
            } else if (sent.equals("Positive")) {
                sentScore = 2;
            } else if (sent.equals("Neutral")) {
                sentScore = 3;

            } else if (sent.equals("Negative")) {
                sentScore =4;
            } else if (sent.equals("Very negative")) {
                sentScore = 5;
            } else {
                System.out.println("Unknown sentiment " + sent);
                sentScore = -1;   //sentinemnt unclear
            }
            avgSent+=sentScore;

            //implement sentiment analysis in this, and then check POS (figure out POS within the sentence)
        }

        avgSent/=document.sentences().size();
       // System.out.println("Expected score: "+score);
        System.out.println("Predicted score: "+avgSent);
    }

    public static boolean unrelated(CoreSentence sentence){

        //TODO: check subject of sentence
        if(sentence.text().matches("in (this|the)(\\s(a-z)*\\s)(movie||film)")){
            return true;
        }

        else if(sentence.text().contains("sequal")){
            System.out.println("IsPlot case 3");
            return true;
        }
        for(int j = 0; j < sentence.tokens().size(); j++){
            CoreLabel token = sentence.tokens().get(j);
            //in the case that the subject of sentence is the film, check if it is describing plot (verb, e.g.: shows, presentes, etc.) or evaluating the film (adj.: e.g.: is ...,
            if(token.word().equals("movie")||token.word().equals("film")){
                System.out.println(sentence.tokens().get(j+1)+" tag: "+sentence.posTags().get(j+1));

                if(sentence.posTags().get(j+1).equals("JJ")||sentence.posTags().get(j+1).equals("JJR")||sentence.posTags().get(j+1).equals("JJS")){
                    return false;
                }else if(sentence.posTags().get(j+1).equals("VB")||sentence.posTags().get(j+1).equals("VBP")||sentence.posTags().get(j+1).equals("VBZ")||sentence.posTags().get(j+1).equals("VBD")||sentence.posTags().get(j+1).equals("VBG")||sentence.posTags().get(j+1).equals("VBN")){
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
