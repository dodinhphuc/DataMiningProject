package com.datamining_project.pre_processing;

import com.datamining_project.constants.Constants;

import weka.core.Instances;

public class Preprocess {
    public Preprocess(){
        this(Constants.INPUT_PATH, Constants.OUTPUT_PATH);
    }

    public Preprocess(String inputFile, String outputFile){
        initializePreprocessing(inputFile, outputFile);
    }

    private void initializePreprocessing(String inputFile, String outputFile){
        Instances data = PreprocessUtils.loadData(inputFile);
        data = PreprocessUtils.removeAttributes(data, 1);
        data = PreprocessUtils.replaceMissingValues(data, 0);
        data = PreprocessUtils.removeOutliers(data);
        data = PreprocessUtils.SMOTE(data, 0);
        data = PreprocessUtils.labelEncoding(data, 0, "0=Blood Donor", 0, "0s=suspect Blood Donor", 1, "1=Hepatitis", 2, "2=Fibrosis", 3, "3=Cirrhosis", 4);
        data = PreprocessUtils.labelEncoding(data, 1, "m", 1, "f", 0);
        PreprocessUtils.displayCorrelationMatrix(data);
        PreprocessUtils.saveData(data, outputFile);
    }
}
