package com.datamining_project.processing.delegate;

import weka.classifiers.Classifier;
import weka.core.Instances;

public class DelegateController {
    private DelegateInterface delegate;
    
    public DelegateController(DelegateInterface delegate){
        this.delegate = delegate;
    }

    public Classifier classifier(Instances data){
        return delegate.classifier(data);
    }
}
