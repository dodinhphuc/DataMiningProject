package com.datamining_project.processing;

import com.datamining_project.constants.Constants;
import com.datamining_project.evaluation.EvaluationUtils;
import com.datamining_project.pre_processing.PreprocessUtils;

import weka.classifiers.Classifier;
import weka.core.Instances;

public class Processing {

    // Constructor that runs the models when the object is instantiated
    public Processing() {
        this(Constants.OUTPUT_PATH_TRAINING);
    }

    public Processing(String inputPath){
        initializeProcessing(inputPath);
    }

    private void initializeProcessing(String inputPath){
        Instances data = PreprocessUtils.loadData(inputPath);
        Classifier decisionTree = ModelUtils.decisionTree(data);
        Classifier naiveBayes = ModelUtils.naiveBayes(data);
        Classifier logistic = ModelUtils.logistic(data);
        Classifier randomforest = ModelUtils.randomForest(data);

        EvaluationUtils.crossValidateModel(decisionTree, data, 10);
        EvaluationUtils.crossValidateModel(naiveBayes, data, 10);
        EvaluationUtils.crossValidateModel(logistic, data, 10);
        EvaluationUtils.crossValidateModel(randomforest, data, 10);
    }
}
