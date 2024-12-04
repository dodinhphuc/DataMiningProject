package com.datamining_project.processing;

import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;  // Decision Tree (J48 is Weka's implementation of C4.5)
import weka.classifiers.functions.Logistic;  // Logistic Regression
import weka.classifiers.bayes.NaiveBayes;  // Naive Bayes
import weka.classifiers.trees.RandomForest;  // Random Forest
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;  // Discretize filter for attributes
import weka.filters.unsupervised.attribute.NumericToNominal;  // Convert numeric class to nominal
import weka.classifiers.Evaluation;

public class ProcessingUtils {

    // Load the dataset from ARFF file (output1.arff)
    public static Instances loadDataset(String path) throws Exception {
        DataSource source = new DataSource(path);
        Instances dataset = source.getDataSet();
        if (dataset.classIndex() == -1)
            dataset.setClassIndex(dataset.numAttributes() - 1); // Set the last attribute as the class
        return dataset;
    }

    // Logistic Regression
public static void runLogisticRegression(Instances data, double ridge) throws Exception {
    long startTime = System.currentTimeMillis();  // Start timer

    // Apply discretization to the attributes (not the class attribute)
    Discretize discretize = new Discretize();
    discretize.setInputFormat(data);
    data = Filter.useFilter(data, discretize);  // Apply the discretization filter to the attributes

    // Convert the class attribute to nominal if it's numeric
    if (data.classAttribute().isNumeric()) {
        NumericToNominal numericToNominal = new NumericToNominal();
        numericToNominal.setAttributeIndices(String.valueOf(data.classIndex() + 1)); // 1-based index
        numericToNominal.setInputFormat(data);
        data = Filter.useFilter(data, numericToNominal);  // Apply the filter to convert class to nominal
        System.out.println("Class attribute is numeric. Converted to nominal...");
    }

    // Logistic Regression
    Logistic logistic = new Logistic();
    logistic.setRidge(ridge); // User-defined ridge parameter
    logistic.buildClassifier(data);
    Evaluation evaluation = new Evaluation(data);
    evaluation.evaluateModel(logistic, data);
    System.out.println("Logistic Regression Evaluation: \n" + evaluation.toSummaryString());

    long endTime = System.currentTimeMillis();  // End timer
    System.out.println("Logistic Regression took " + (endTime - startTime) + " milliseconds to build and evaluate.");
}

// Naive Bayes
public static void runNaiveBayes(Instances data, boolean useKernelEstimator) throws Exception {
    long startTime = System.currentTimeMillis();  // Start timer

    // Apply discretization to the attributes (not the class attribute)
    Discretize discretize = new Discretize();
    discretize.setInputFormat(data);
    data = Filter.useFilter(data, discretize);  // Apply the discretization filter to the attributes

    // Convert the class attribute to nominal if it's numeric
    if (data.classAttribute().isNumeric()) {
        NumericToNominal numericToNominal = new NumericToNominal();
        numericToNominal.setAttributeIndices(String.valueOf(data.classIndex() + 1)); // 1-based index
        numericToNominal.setInputFormat(data);
        data = Filter.useFilter(data, numericToNominal);  // Apply the filter to convert class to nominal
        System.out.println("Class attribute is numeric. Converted to nominal...");
    }

    // Naive Bayes
    NaiveBayes naiveBayes = new NaiveBayes();
    naiveBayes.setOptions(new String[] {
        useKernelEstimator ? "-K" : "" // Use kernel estimator if true
    });
    naiveBayes.buildClassifier(data);
    Evaluation evaluation = new Evaluation(data);
    evaluation.evaluateModel(naiveBayes, data);
    System.out.println("Naive Bayes Evaluation: \n" + evaluation.toSummaryString());

    long endTime = System.currentTimeMillis();  // End timer
    System.out.println("Naive Bayes took " + (endTime - startTime) + " milliseconds to build and evaluate.");
}

// Decision Tree (J48)
public static void runDecisionTree(Instances data, double confidenceFactor, int minNumObj) throws Exception {
    long startTime = System.currentTimeMillis();  // Start timer

    // Apply discretization to the attributes (not the class attribute)
    Discretize discretize = new Discretize();
    discretize.setInputFormat(data);
    data = Filter.useFilter(data, discretize);  // Apply the discretization filter to the attributes

    // Convert the class attribute to nominal if it's numeric
    if (data.classAttribute().isNumeric()) {
        NumericToNominal numericToNominal = new NumericToNominal();
        numericToNominal.setAttributeIndices(String.valueOf(data.classIndex() + 1)); // 1-based index
        numericToNominal.setInputFormat(data);
        data = Filter.useFilter(data, numericToNominal);  // Apply the filter to convert class to nominal
        System.out.println("Class attribute is numeric. Converted to nominal...");
    }

    // Decision Tree (J48)
    J48 decisionTree = new J48();
    decisionTree.setOptions(new String[] {
        "-C", String.valueOf(confidenceFactor), // Confidence factor
        "-M", String.valueOf(minNumObj)        // Minimum number of objects per leaf
    });
    decisionTree.buildClassifier(data);
    Evaluation evaluation = new Evaluation(data);
    evaluation.evaluateModel(decisionTree, data);
    System.out.println("Decision Tree Evaluation: \n" + evaluation.toSummaryString());

    long endTime = System.currentTimeMillis();  // End timer
    System.out.println("Decision Tree (J48) took " + (endTime - startTime) + " milliseconds to build and evaluate.");
}

// Random Forest
public static void runRandomForest(Instances data, int numTrees, int maxDepth) throws Exception {
    long startTime = System.currentTimeMillis();  // Start timer

    // Apply discretization to the attributes (not the class attribute)
    Discretize discretize = new Discretize();
    discretize.setInputFormat(data);
    data = Filter.useFilter(data, discretize);  // Apply the discretization filter to the attributes

    // Convert the class attribute to nominal if it's numeric
    if (data.classAttribute().isNumeric()) {
        NumericToNominal numericToNominal = new NumericToNominal();
        numericToNominal.setAttributeIndices(String.valueOf(data.classIndex() + 1)); // 1-based index
        numericToNominal.setInputFormat(data);
        data = Filter.useFilter(data, numericToNominal);  // Apply the filter to convert class to nominal
        System.out.println("Class attribute is numeric. Converted to nominal...");
    }

    // Random Forest
    RandomForest randomForest = new RandomForest();
    randomForest.setOptions(new String[] {
        "-I", String.valueOf(numTrees),  // Number of trees
        "-depth", String.valueOf(maxDepth) // Maximum depth
    });

    // Perform 10-fold cross-validation
    Evaluation evaluation = new Evaluation(data);
    evaluation.crossValidateModel(randomForest, data, 10, new java.util.Random(1));  // 10-fold cross-validation

    // Output the evaluation results
    System.out.println("Random Forest Evaluation: \n" + evaluation.toSummaryString());

    long endTime = System.currentTimeMillis();  // End timer
    System.out.println("Random Forest took " + (endTime - startTime) + " milliseconds to build and evaluate.");
}

}
