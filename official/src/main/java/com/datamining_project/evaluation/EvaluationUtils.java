package com.datamining_project.evaluation;

import java.util.Random;

import com.datamining_project.processing.delegate.DelegateController;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;

public class EvaluationUtils {
    public static void crossValidateModel(Classifier classifier, Instances data, int fold){
        try{
            if (data.classIndex()<0){
                data.setClassIndex(data.numAttributes()-1);
            }
            Evaluation evaluation = new Evaluation(data);
            evaluation.crossValidateModel(classifier, data, 10, new Random(1));
            System.out.println(classifier.toString() + "\nEvaluation: \n" + evaluation.toSummaryString());
        } catch (Exception e){
            System.out.println("Error in evaluation!");
            System.err.println(e);
        }
    }

    public static void train_testValidate(Classifier classifier, Instances trainingSet, Instances testingSet){
        try {
            if (trainingSet.classIndex()<0){
                trainingSet.setClassIndex(trainingSet.numAttributes()-1);
            }
            if (testingSet.classIndex()<0){
                testingSet.setClassIndex(testingSet.numAttributes()-1);
            }
            Evaluation evaluation = new Evaluation(trainingSet);
            evaluation.evaluateModel(classifier, testingSet);
            System.out.println(classifier.toString() + "\nEvaluation: \n" + evaluation.toSummaryString());
            double z = 1.65;
            double f = (double) evaluation.pctCorrect() / 100;
            int n = testingSet.size();
            double lowerBound = (f + z*z/(2*n) - z*Math.sqrt(f/n - f*f/n + z*z/(4*n*n)))/(1+z*z/n);
            double upperBound = (f + z*z/(2*n) + z*Math.sqrt(f/n - f*f/n + z*z/(4*n*n)))/(1+z*z/n);
            System.out.printf("\nWith %3s confidence, the precision is in the range (in percentage) [%6.3f, %6.3f].\n", "90%",lowerBound*100, upperBound*100);
        } catch (Exception e){
            System.out.println("Error in evaluation!");
            System.err.println(e);
        }
    }

    public static void trainingModelTime(DelegateController delegateController, Instances data){
        long startTime = System.currentTimeMillis();
        if (data.classIndex()<0){
            data.setClassIndex(data.numAttributes()-1);
        }
        Classifier classifier = delegateController.classifier(data);
        long endTime = System.currentTimeMillis();
        System.out.println("\nTakes " + (endTime - startTime) + " (milisecs) to build!");
        System.out.println("____________________________________________________");
    }
}
