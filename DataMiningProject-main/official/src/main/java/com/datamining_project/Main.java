package com.datamining_project;

import com.datamining_project.pre_processing.Preprocess;
import com.datamining_project.processing.Processing;

public class Main {

    public static void main(String[] args) {
        
    
        new Preprocess();  // Calls the preprocessing actions
        new Processing();  // Runs the models with the hardcoded parameters

    
    }
}
