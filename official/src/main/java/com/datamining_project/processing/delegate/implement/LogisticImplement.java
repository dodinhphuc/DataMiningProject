package com.datamining_project.processing.delegate.implement;

import com.datamining_project.processing.ModelUtils;
import com.datamining_project.processing.delegate.DelegateInterface;

import weka.classifiers.Classifier;
import weka.core.Instances;

public class LogisticImplement implements DelegateInterface{

    @Override
    public Classifier classifier(Instances data) {
        return ModelUtils.logistic(data);
    }
}
