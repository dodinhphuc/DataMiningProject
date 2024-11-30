package com.datamining_project.processing;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.J48;
import weka.classifiers.functions.Logistic;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.Evaluation;
import java.util.Random;

public class ClassifyInstance {
    public static void main(String[] args) throws Exception {

        // Load dataset
        DataSource dataSource = new DataSource("../data/pre_processing/output1.arff");
        Instances dataset = dataSource.getDataSet();

        // Set the last attribute as the class
        dataset.setClassIndex(dataset.numAttributes() - 1);

        // Step 2: Using Logistic Regression, J48, NaiveBayes, and Random Forest classifiers

        // Logistic Regression classifier
        Logistic logistic = new Logistic();
        logistic.buildClassifier(dataset);
        System.out.println("Logistic Regression Classifier built.");

        // J48 classifier (Decision Tree)
        J48 j48 = new J48();
        j48.buildClassifier(dataset);
        System.out.println("J48 Classifier built.");

        // NaiveBayes classifier
        NaiveBayes nb = new NaiveBayes();
        nb.buildClassifier(dataset);
        System.out.println("NaiveBayes Classifier built.");

        // Random Forest classifier
        RandomForest rf = new RandomForest();
        rf.buildClassifier(dataset);
        System.out.println("Random Forest Classifier built.");

        // Step 3: Evaluate the models
        evaluateModel(logistic, dataset);
        evaluateModel(j48, dataset);
        evaluateModel(nb, dataset);
        evaluateModel(rf, dataset);
    }

    // Method to evaluate a model using 10-fold cross-validation
    public static void evaluateModel(weka.classifiers.Classifier model, Instances dataset) throws Exception {
        Evaluation evaluation = new Evaluation(dataset);
        evaluation.crossValidateModel(model, dataset, 10, new Random(1));
        System.out.println("Model Evaluation: " + model.getClass().getName());
        System.out.println("Correctly Classified Instances: " + evaluation.pctCorrect() + "%");
        System.out.println("Incorrectly Classified Instances: " + evaluation.pctIncorrect() + "%");
        System.out.println("Confusion Matrix: \n" + evaluation.toMatrixString());
        System.out.println("====================================");
    }
}
