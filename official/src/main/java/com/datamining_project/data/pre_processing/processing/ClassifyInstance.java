package com.datamining_project.data.pre_processing.processing; 
import weka.core.Instance; 
import weka.core.Instances; 
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.J48;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation; 
import weka.core.DenseInstance;

public class ClassifyInstance {
    public static void main(String[] args) throws Exception { 

        // load training dataset
        DataSource dataSource = new DataSource("official\\src\\main\\java\\com\\datamining_project\\data\\pre_processing\\output1.arff");
        Instances dataset = dataSource.getDataSet();  

        // set class index to the last attribute 
        dataset.setClassIndex(dataset.numAttributes() - 1);

        // get the value of class
        int numClasses = dataset.numClasses();
        for (int i = 0; i < numClasses; i++) { 
            String classValue = dataset.classAttribute().value(i); 
            System.out.println("Class value " + i + " is " + classValue);
        }

        // create classifier
        NaiveBayes nb = new NaiveBayes();
        nb.buildClassifier(dataset);  

        // J48 
        J48 tree = new J48();
        tree.buildClassifier(dataset);
        System.out.println("J48 Tree Model:");
        System.out.println(tree);

        // Evaluation
        Evaluation evaluation = new Evaluation(dataset);
        evaluation.crossValidateModel(tree, dataset, 10, new java.util.Random(1));  
        evaluation.crossValidateModel(nb, dataset, 10, new java.util.Random(1));
        System.out.println("Evaluation results: ");
        System.out.println(evaluation.toSummaryString());

        // Loop through the dataset and make predictions on each instance
        System.out.println("Making predictions on dataset instances:");
        for (int i = 0; i < dataset.numInstances(); i++) {
            Instance instance = dataset.instance(i);
            double predictedClass = nb.classifyInstance(instance); 
            double predictedClass2 = tree.classifyInstance(instance);
            String predictedClassLabel = dataset.classAttribute().value((int) predictedClass); 
            String predictedClassLabel2 = dataset.classAttribute().value((int) predictedClass2);  // J48 tree prediction
            System.out.println("Instance " + i + ": Predicted Class = " + predictedClassLabel);
        }

        // Example of classifying a new instance manually:
        System.out.println("Classifying a new instance:");
        // Create a new instance with the same number of attributes as the dataset
        Instance newInstance = new DenseInstance(dataset.numAttributes());
        newInstance.setValue(0, 32);   // Age
        newInstance.setValue(1, 38.5); // ALB
        newInstance.setValue(2, 52.5); // ALP
        newInstance.setValue(3, 7.7);  // ALT
        newInstance.setValue(4, 22.1); // AST
        newInstance.setValue(5, 7.5);  // BIL
        newInstance.setValue(6, 6.93); // CHE
        newInstance.setValue(7, 3.23); // CHOL
        newInstance.setValue(8, 106);  // CREA
        newInstance.setValue(9, 12.1); // GGT
        newInstance.setValue(10, 69);  // PROT
        newInstance.setValue(11, 0);   // Category_
        newInstance.setValue(12, 1);   // Sex_
 
        // Set class index for the new instance
        newInstance.setDataset(dataset);
 
        // Predict the class of the new instance
        double predictedClassNew = nb.classifyInstance(newInstance);   
        double predictedClassNew2 = tree.classifyInstance(newInstance);
        String predictedClassLabelNew = dataset.classAttribute().value((int) predictedClassNew); 
        System.out.println("New Instance Predicted Class: " + predictedClassLabelNew);
    }
}
