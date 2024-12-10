package com.datamining_project;

import org.omg.CORBA.portable.Delegate;

import com.datamining_project.constants.Constants;
import com.datamining_project.evaluation.Evaluation;
import com.datamining_project.evaluation.EvaluationUtils;
import com.datamining_project.pre_processing.Preprocess;
import com.datamining_project.pre_processing.PreprocessUtils;
import com.datamining_project.processing.ModelUtils;
import com.datamining_project.processing.Processing;

import com.datamining_project.processing.delegate.DelegateController;
import com.datamining_project.processing.delegate.implement.DecisionTreeImplement;

import weka.classifiers.Classifier;
import weka.core.Instances;



public class Main {
    public static void main(String[] args) {
        new Preprocess();
        
        //new Processing();
        //new Evaluation();


        // Instances data = PreprocessUtils.loadData(Constants.INPUT_PATH);
        // data = PreprocessUtils.removeAttributes(data, 1);
        // data = PreprocessUtils.replaceMissingValues(data, 0); 
        // data = PreprocessUtils.removeOutliers(data);
        // data = PreprocessUtils.labelEncoding(data, 2, "m", 1, "f", 0);
        // data = PreprocessUtils.SMOTE(data, 0);
        // PreprocessUtils.attributeStats(data, 0);
        //Instances test = PreprocessUtils.loadData(Constants.OUTPUT_PATH_VALIDATING);
        //data = PreprocessUtils.discretize(data, 20, 0, 1);
        //data = PreprocessUtils.numericToNominal(data, 12);
        //PreprocessUtils.saveData(data, Constants.TEST);
        //Classifier classifier = ModelUtils.decisionTree(data);
        //DelegateController delegateModel = new DelegateController(new DecisionTreeImplement());
        //EvaluationUtils.crossValidateModel(classifier, data, 10);
        //EvaluationUtils.train_testValidate(classifier, data, test);
        //EvaluationUtils.trainingModelTime(delegateModel, data);
    }
}