package com.datamining_project.pre_processing;

import com.datamining_project.constants.Constants;

import weka.core.Instances;

public class Preprocess {
    public Preprocess(){
        initializePreprocessing();
    }

    private void initializePreprocessing(){
        Instances data = PreprocessUtils.loadData(Constants.INPUT_PATH);
        data = PreprocessUtils.removeAttributes(data, 1);
        data = PreprocessUtils.replaceMissingValues(data, 0);
        data = PreprocessUtils.removeOutliers(data);
        data = PreprocessUtils.SMOTE(data, 0);
        data = PreprocessUtils.labelEncoding(data, 0, "0=Blood Donor", 0, "0s=suspect Blood Donor", 1, "1=Hepatitis", 2, "2=Fibrosis", 3, "3=Cirrhosis", 4);
        data = PreprocessUtils.labelEncoding(data, 1, "m", 1, "f", 0);
        PreprocessUtils.displayCorrelationMatrix(data);
        PreprocessUtils.saveData(data, Constants.OUTPUT_PATH);
    }
}
