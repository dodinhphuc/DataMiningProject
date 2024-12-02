package com.datamining_project.processing;

import weka.core.Instances;
import com.datamining_project.processing.ProcessingUtils;

public class Processing {

    // Constructor that runs the models when the object is instantiated
    public Processing() {
        try {
            // Hardcoded parameters
            String datasetPath = "C:\\Users\\bin\\Downloads\\DataMiningProject-main\\DataMiningProject-main\\official\\src\\main\\java\\com\\datamining_project\\data\\pre_processing\\output1.arff";  
            double ridge = 1.0;  // Ridge parameter for Logistic Regression
            boolean useKernelEstimator = true;  // Use Kernel Estimator for Naive Bayes
            double confidenceFactor = 0.25;  // Confidence factor for Decision Tree
            int minNumObj = 2;  // Minimum number of objects per leaf for Decision Tree
            int numTrees = 100;  // Number of trees for Random Forest
            int maxDepth = 10;  // Maximum depth for Random Forest

            // Load the dataset
            Instances data = ProcessingUtils.loadDataset(datasetPath);

            // Run Logistic Regression
            System.out.println("Running Logistic Regression...");
            ProcessingUtils.runLogisticRegression(data, ridge);

            // Run Naive Bayes
            System.out.println("\nRunning Naive Bayes...");
            ProcessingUtils.runNaiveBayes(data, useKernelEstimator);

            // Run Decision Tree (J48)
            System.out.println("\nRunning Decision Tree (J48)...");
            ProcessingUtils.runDecisionTree(data, confidenceFactor, minNumObj);

            // Run Random Forest
            System.out.println("\nRunning Random Forest...");
            ProcessingUtils.runRandomForest(data, numTrees, maxDepth);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
