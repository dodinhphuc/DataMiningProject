package com.datamining_project.processing.delegate;

import weka.classifiers.Classifier;
import weka.core.Instances;

public interface DelegateInterface {
    public Classifier classifier(Instances data);
}
