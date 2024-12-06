package com.datamining_project.processing;

import com.datamining_project.pre_processing.PreprocessUtils;

import weka.classifiers.functions.Logistic;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;


import weka.core.Instances; 

public class ModelUtils {
    @SuppressWarnings("finally")
    public static Logistic logistic(Instances data){
        Logistic logistic = new Logistic();
        try{
            if (data.classIndex()<0){
                data.setClassIndex(data.numAttributes()-1);
            }
            logistic.buildClassifier(data);
            System.out.println("Successfully in building Logistic model!");
        }catch (Exception e) {
            System.out.println("Error in building Logistic model!");
            System.err.println(e);
        }finally {
            return logistic;
        }
    }

    @SuppressWarnings("finally")
    public static Logistic logistic(Instances data,  double ridge,  int binNumber_discretizing, Object... attributeNumbers_discretizing){
        Logistic logistic = new Logistic();
        Instances new_data = new Instances(data);
        try{
            if (new_data.classIndex()<0){
                new_data.setClassIndex(new_data.numAttributes()-1);
            }
            if (binNumber_discretizing>0){
                new_data = PreprocessUtils.discretize(new_data, binNumber_discretizing, attributeNumbers_discretizing);
                new_data = PreprocessUtils.discretize(new_data, 2, 11);
            }
            if (ridge>0){
                logistic.setRidge(ridge);
            }
            
            logistic.buildClassifier(new_data);
            System.out.println("Successfully in building Logistic model!");
        }catch (Exception e) {
            System.out.println("Error in building Logistic model!");
            System.err.println(e);
        }finally {
            return logistic;
        }
    }

    @SuppressWarnings("finally")
    public static NaiveBayes naiveBayes(Instances data){
        NaiveBayes naiveBayes = new NaiveBayes();
        try{
            if (data.classIndex()<0){
                data.setClassIndex(data.numAttributes()-1);
            }
            naiveBayes.buildClassifier(data);
            System.out.println("Successfully in building Naive Bayes model!");
        }catch (Exception e) {
            System.out.println("Error in building Naive Bayes model!");
            System.err.println(e);
        }finally {
            return naiveBayes;
        }
    }

    @SuppressWarnings("finally")
    public static NaiveBayes naiveBayes(Instances data, int binNumber_discretizing, Object... attributeNumbers_discretizing){
        NaiveBayes naiveBayes = new NaiveBayes();
        Instances new_data = new Instances(data);
        try{
            if (new_data.classIndex()<0){
                new_data.setClassIndex(new_data.numAttributes()-1);
            }
            if (binNumber_discretizing>0){
                new_data = PreprocessUtils.discretize(new_data, binNumber_discretizing, attributeNumbers_discretizing);
                new_data = PreprocessUtils.discretize(new_data, 2, 11);
            }
            naiveBayes.buildClassifier(new_data);
            System.out.println("Successfully in building naive Bayes model!");
        }catch (Exception e) {
            System.out.println("Error in building naive Bayes model!");
            System.err.println(e);
        }finally {
            return naiveBayes;
        }
    }

    @SuppressWarnings("finally")
    public static J48 decisionTree(Instances data){
        J48 decisionTree = new J48();
        try{
            if (data.classIndex()<0){
                data.setClassIndex(data.numAttributes()-1);
            }
            decisionTree.buildClassifier(data);
            System.out.println("Successfully in building decision tree model!");
        }catch (Exception e) {
            System.out.println("Error in building decision tree model!");
            System.err.println(e);
        }finally {
            return decisionTree;
        }
    }

    @SuppressWarnings("finally")
    public static J48 decisionTree(Instances data, int minNumObj, float confidenceFactor, int binNumber_discretizing, Object... attributeNumbers_discretizing){
        J48 decisionTree = new J48();   
        Instances new_data = new Instances(data);
        try{
            if (new_data.classIndex()<0){
                new_data.setClassIndex(new_data.numAttributes()-1);
            }
            if (binNumber_discretizing>0){
                new_data = PreprocessUtils.discretize(new_data, binNumber_discretizing, attributeNumbers_discretizing);
                new_data = PreprocessUtils.discretize(new_data, 2, 11);
            }
            if (minNumObj>0){
                decisionTree.setMinNumObj(minNumObj);
            }
            if (confidenceFactor>0){
                decisionTree.setConfidenceFactor(confidenceFactor);
            }
            decisionTree.buildClassifier(new_data);
            System.out.println("Successfully in building decision tree model!");
        }catch (Exception e) {
            System.out.println("Error in building decision tree model!");
            System.err.println(e);
        }finally {
            return decisionTree;
        }
    }

    @SuppressWarnings("finally")
    public static RandomForest randomForest(Instances data){
        RandomForest randomForest = new RandomForest();
        try{
            if (data.classIndex()<0){
                data.setClassIndex(data.numAttributes()-1);
            }
            randomForest.buildClassifier(data);
            System.out.println("Successfully in building Random Forest model!");
        }catch (Exception e) {
            System.out.println("Error in building Random Forest model!");
            System.err.println(e);
        }finally {
            return randomForest;
        }
    }

    @SuppressWarnings("finally")
    public static RandomForest randomForest(Instances data, int numTrees, int maxDepth, int binNumber_discretizing, Object... attributeNumbers_discretizing){
        RandomForest randomForest = new RandomForest();
        Instances new_data = new Instances(data);
        try{
            if (new_data.classIndex()<0){
                new_data.setClassIndex(new_data.numAttributes()-1);
            }
            if (binNumber_discretizing>0){
                new_data = PreprocessUtils.discretize(new_data, binNumber_discretizing, attributeNumbers_discretizing);
                new_data = PreprocessUtils.discretize(new_data, 2, 11);
            }
            if (numTrees>0){
                randomForest.setNumIterations(numTrees);
            }
            if (maxDepth>0){
                randomForest.setMaxDepth(maxDepth);
            }
            randomForest.buildClassifier(new_data);
            System.out.println("Successfully in building Random Forest model!");
        }catch (Exception e) {
            System.out.println("Error in building Random Forest model!");
            System.err.println(e);
        }finally {
            return randomForest;
        }
    }

}
