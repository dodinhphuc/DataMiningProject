package com.datamining_project.pre_processing;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.converters.ArffSaver;
import weka.core.Utils;

import weka.core.AttributeStats;
import weka.core.DenseInstance;
import weka.core.Attribute;
import weka.core.Instance;

import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.supervised.instance.SMOTE;
import weka.filters.unsupervised.attribute.InterquartileRange;
import weka.filters.unsupervised.instance.RemoveWithValues;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

import weka.attributeSelection.PrincipalComponents;

public class PreprocessUtils {

    //load the data file from the path
    //use the path in the Constant.java
    @SuppressWarnings("finally")
    public static Instances loadData(String path){
        Instances data = null;
        try{
            DataSource loader = new DataSource(path);
            data = loader.getDataSet();
            System.out.println("Load file successfully");
        } catch (Exception e){
            System.err.println("Error in loading file!"); 
            System.err.println(e);              
        } finally {
            return data;
        }
    }

    //save data to the path with arff type
    //use the path in the Constants.java
    public static void saveData(Instances data, String path){
        try{
            ArffSaver saver = new ArffSaver();
            saver.setInstances(data);
            saver.setFile(new File(path));
            saver.writeBatch();
            System.out.println("Save file successfully!");
        } catch (Exception e){
            System.err.println("Error in saving file!");
            System.err.println(e);   
        }
    }

    // for example, if you want to delete attribute 1 and 2 from the data
    // call removeAttributes(data, 1, 2)
    // first attribute has index 1 :))
    @SuppressWarnings("finally")
    public static Instances removeAttributes(Instances data, Object... objects){
        Instances instances = null;
        try{
            Remove remove = new Remove();
            objects = sortingDecreasing(objects);
            for (int i=0; i<objects.length; ++i){
                String split_opt = "-R";
                split_opt = split_opt + " " + objects[i];
                remove.setOptions(Utils.splitOptions(split_opt));
                if (i == 0) {
                    remove.setInputFormat(data);
                    instances = Filter.useFilter(data, remove);
                } else {
                    remove.setInputFormat(instances);
                    instances = Filter.useFilter(instances, remove);
                }
                System.out.println("Successfully remove attribute " + objects[i] + "!");
            }
        } catch (Exception e){
            System.err.println("Error in removing attribute");
            System.err.println(e);   
        } finally {
            return instances;
        }
    }
    
    //apply SMOTE for the data with specific class index
    //note that the attribute index of this method start from 0
    @SuppressWarnings("finally")
    public static Instances SMOTE(Instances data, int classIndex){
        Instances instances = null;
        try{
            SMOTE smote = new SMOTE();
            String[] opt = new String[]{"-P", "20.0"};
            smote.setOptions(opt);
            data.setClassIndex(classIndex);
            smote.setInputFormat(data); 
            instances = Filter.useFilter(data, smote);
            while (!checkEqual(instances, classIndex)) {
                instances.setClassIndex(classIndex);
                smote.setInputFormat(instances);
                instances = Filter.useFilter(instances, smote);
            }
            System.out.println("SMOTE successfully!");
        } catch (Exception e){
            System.err.println("Error in SMOTE!");
            System.err.println(e);
        } finally {
            return instances;
        }
    }

    //remove outlier for the data
    //outlier factor is set by 12.0
    @SuppressWarnings("finally")
    public static Instances removeOutliers(Instances data){
        Instances instances = null;
        try {   
            InterquartileRange interquartileRange = new InterquartileRange();
            interquartileRange.setInputFormat(data);
            interquartileRange.setExtremeValuesFactor(15.0);
            interquartileRange.setOutlierFactor(12.0);
            interquartileRange.setExtremeValuesAsOutliers(true);
            instances = Filter.useFilter(data, interquartileRange);

            RemoveWithValues removeWithValues = new RemoveWithValues();
            String[] opt = new String[] {"-C", ((Integer) (data.numAttributes() + 1)).toString(), "-L", "last"};
            removeWithValues.setOptions(opt);
            removeWithValues.setInputFormat(instances);
            instances = Filter.useFilter(instances, removeWithValues);

            instances = removeAttributes(instances, data.numAttributes()+1);

            instances = removeAttributes(instances, data.numAttributes()+1);
            System.out.println("Remove outliers successfully!");
        } catch (Exception e){
            System.out.println("Error in removing outliers!");
            System.err.println(e);
        } finally {
            return instances;
        }
    }

    public static void displayCorrelationMatrix(Instances instances){
        try{
            PrincipalComponents principalComponents = new PrincipalComponents();
            principalComponents.buildEvaluator(instances);
            double[][] crorelation_matrix = principalComponents.getCorrelationMatrix();
            System.out.printf("%8s", "");
            for (int j=0; j<crorelation_matrix[1].length; j++){
                System.out.printf("%10s", instances.attribute(j).name());
            }
            System.out.println();
            for (int i=0; i<crorelation_matrix.length; i++){
                System.out.printf("%8s", instances.attribute(i).name());
                for (int j=0; j<crorelation_matrix[i].length; j++){
                    System.out.printf("%10.3f", crorelation_matrix[i][j]);
                }
                System.out.println();
            }
        } catch (Exception e){
            System.out.println("Error in loading correlation matrix!");
            System.err.println(e);
        }
    }

    @SuppressWarnings("finally")
    public static Instances labelEncoding(Instances data, int attributeNumber, Object... keyAndValuePairs){
        Instances instances = null;
        HashMap<String, Integer> encoder = new HashMap<>();
        try{
            for (int i=0; i<keyAndValuePairs.length; i+=2){
                encoder.put((String) keyAndValuePairs[i], (Integer) keyAndValuePairs[i+1]);
            }
            data.setClassIndex(attributeNumber);
            Attribute attribute = new Attribute(data.attribute(attributeNumber).name()+"_");
            Instances instances2 = new Instances("class", new ArrayList<Attribute>(){{add(attribute);}}, 1000000);
            for (int i=0; i<data.size(); ++i){
                DenseInstance instance = new DenseInstance(1); 
                Instance instance2 = data.instance(i);
                double cV = instance2.classValue();
                String key = instance2.classAttribute().value((int) cV);
                Integer value = encoder.get(key);
                instance.setValue(0, value);
                instances2.add(instance);
            }
            instances = Instances.mergeInstances(data, instances2);
            instances = removeAttributes(instances, attributeNumber+1);
            System.out.println("Label encoding successfully!");
        } catch (Exception e){
            System.out.println("Error in encoding!");
            System.err.println(e);
        } finally {
            return instances;
        }
    }

    @SuppressWarnings("finally")
    public static Instances replaceMissingValues(Instances data, int classIndex){
        Instances instances = null;
        List<String> attributeDistinctValues = new ArrayList<>();
        try {
            instances = new Instances(data, 1000000);
            Instances instances2 = null;
            data.setClassIndex(classIndex);
            for (int i=0; i<data.classAttribute().numValues(); i++){
                attributeDistinctValues.add(data.classAttribute().value(i));
            }
            Instance instance = null;
            ReplaceMissingValues replaceMissingValues = new ReplaceMissingValues();
            for (String value : attributeDistinctValues){
                instances2 = new Instances(data, 1000000);
                for (int i=0; i<data.size(); ++i){
                    instance = data.get(i);
                    if (instance.classAttribute().value((int) instance.classValue()).equals(value)) instances2.add(instance);
                }
                replaceMissingValues.setInputFormat(instances2);
                instances2 = Filter.useFilter(instances2, replaceMissingValues);
                for (int i=0; i<instances2.size(); i++){
                    instances.add(instances2.instance(i));
                }
            }
            System.out.println("Handle missing values successfully!");
        } catch (Exception e){
            System.out.println("Error in handling missing values!");
            System.err.println(e);
        } finally {
            return instances;
        }
    }

/////////////////////////////////////////////////////////////////////////////
/// 
/// 
/// This section is for private method
    private static Object[] sortingDecreasing(Object... objects){
        for (int i=1; i<objects.length; ++i){
            int k = i;
            Object temp = objects[i];
            while ((k>0) && ((Integer) temp)>((Integer) objects[--k])){
                objects[k+1] = objects[k]; 
                objects[k] = temp;
            }
        }
        return objects;
    }

    private static boolean checkEqual(Instances data, int classIndex){
        AttributeStats attributeStats = data.attributeStats(classIndex);
        int[] count = attributeStats.nominalCounts;
        int min = count[0];
        int max = count[0];
        for (int i=0; i<count.length; ++i){
            if (min>count[i]) min = count[i];
            if (max<count[i]) max = count[i];
        }
        float k =(float) (max-min)/max;
        if (k < 0.19) return true;
        else return false;
    }
}
