package com.datamining_project.evaluation;

import com.datamining_project.constants.Constants;
import com.datamining_project.pre_processing.PreprocessUtils;
import com.datamining_project.processing.ModelUtils;
import com.datamining_project.processing.delegate.DelegateController;
import com.datamining_project.processing.delegate.implement.DecisionTreeImplement;
import com.datamining_project.processing.delegate.implement.LogisticImplement;
import com.datamining_project.processing.delegate.implement.NaiveBayesImplement;
import com.datamining_project.processing.delegate.implement.RandomForestImplement;

import weka.classifiers.Classifier;
import weka.core.Instances;

public class Evaluation {
    public Evaluation(){
        this(Constants.OUTPUT_PATH_TRAINING, Constants.OUTPUT_PATH_VALIDATING);
    }

    public Evaluation(String inputPath1, String inputPath2){
        initializeEvaluation(inputPath1, inputPath2);
    }

    private void initializeEvaluation(String inputPath1, String inputPath2){
        Instances training = PreprocessUtils.loadData(inputPath1);
        Instances testing = PreprocessUtils.loadData(inputPath2);

        Classifier decisionTree = ModelUtils.decisionTree(training);
        Classifier naiveBayes = ModelUtils.naiveBayes(training);
        Classifier logistic = ModelUtils.logistic(training);
        Classifier randomforest = ModelUtils.randomForest(training);

        DelegateController delegateDecisionTree = new DelegateController(new DecisionTreeImplement());
        DelegateController delegateNaiveBayes = new DelegateController(new NaiveBayesImplement());
        DelegateController delegateLogistic = new DelegateController(new LogisticImplement());
        DelegateController delegateRandomForest = new DelegateController(new RandomForestImplement());

        EvaluationUtils.train_testValidate(decisionTree, training, testing);
        EvaluationUtils.trainingModelTime(delegateDecisionTree, training);

        EvaluationUtils.train_testValidate(naiveBayes, training, testing);
        EvaluationUtils.trainingModelTime(delegateNaiveBayes, training);

        EvaluationUtils.train_testValidate(logistic, training, testing);
        EvaluationUtils.trainingModelTime(delegateLogistic, training);

        EvaluationUtils.train_testValidate(randomforest, training, testing);
        EvaluationUtils.trainingModelTime(delegateRandomForest, training);
    }
}
