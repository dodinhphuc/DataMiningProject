package com.datamining_project.pre_processing;

import com.datamining_project.constants.Constants;

import weka.core.Instances;

public class Preprocess {
    public Preprocess(){
        this(Constants.INPUT_PATH, Constants.OUTPUT_PATH_TRAINING, Constants.OUTPUT_PATH_VALIDATING);
    }

    public Preprocess(String inputFile, String outputPathForTraining, String outputPathForValidating){
        initializePreprocessing(inputFile, outputPathForTraining, outputPathForValidating);
    }

    private void initializePreprocessing(String inputFile, String outputPathForTraining, String outputPathForValidating){
        Instances data = PreprocessUtils.loadData(inputFile);
        data = PreprocessUtils.removeAttributes(data, 1);
        data = PreprocessUtils.replaceMissingValues(data, 0);
        data = PreprocessUtils.removeOutliers(data);
        
        PreprocessUtils.splitData(data, 80, Constants.OUTPUT_PATH, Constants.OUTPUT_PATH_2);
        
        Instances data1 = PreprocessUtils.loadData(Constants.OUTPUT_PATH);
        data1 = PreprocessUtils.labelEncoding(data1, 2, "m", 1, "f", 0);
        data1 = PreprocessUtils.SMOTE(data1, 0);
        data1 = PreprocessUtils.labelEncoding(data1, 0, "0=Blood Donor", 0, "0s=suspect Blood Donor", 1, "1=Hepatitis", 2, "2=Fibrosis", 3, "3=Cirrhosis", 4);
        PreprocessUtils.displayCorrelationMatrix(data1);
        data1 = PreprocessUtils.numericToNominal(data1, data1.numAttributes()-1);
        PreprocessUtils.saveData(data1, outputPathForTraining);
        
        Instances data2 = PreprocessUtils.loadData(Constants.OUTPUT_PATH_2);
        data2 = PreprocessUtils.labelEncoding(data2, 2, "m", 1, "f", 0);
        data2 = PreprocessUtils.labelEncoding(data2, 0, "0=Blood Donor", 0, "0s=suspect Blood Donor", 1, "1=Hepatitis", 2, "2=Fibrosis", 3, "3=Cirrhosis", 4);
        data2 = PreprocessUtils.numericToNominal(data2, data2.numAttributes()-1);
        PreprocessUtils.saveData(data2, outputPathForValidating);
    }
}
