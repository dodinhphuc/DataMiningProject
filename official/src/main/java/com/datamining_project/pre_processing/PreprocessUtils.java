package com.datamining_project.pre_processing;

import java.io.File;
import java.text.AttributedCharacterIterator.Attribute;

import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;

import weka.core.AttributeStats;

import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.supervised.instance.SMOTE;

public class PreprocessUtils {
    @SuppressWarnings("finally")
    public static Instances loadCSVData(String path){
        Instances data = null;
        try{
            CSVLoader loader = new CSVLoader();
            loader.setSource(new File(path));
            data = loader.getDataSet();
            System.out.println(data.toString());
        } catch (Exception e){
            System.err.println("Error in loading file!");
            System.err.println(e);            
        } finally {
            return data;
        }
    }

    @SuppressWarnings("finally")
    public static Instances loadArffData(String path){
        Instances data = null;
        try{
            ArffLoader loader = new ArffLoader();
            loader.setSource(new File(path));
            data = loader.getDataSet();
            System.out.println(data.toString());
        } catch (Exception e){
            System.err.println("Error in loading file!"); 
            System.err.println(e);              
        } finally {
            return data;
        }
    }

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

    // eg. listAttributes = {"-R", "1", "2"}
    @SuppressWarnings("finally")
    public static Instances removeAttributes(Instances data, String[] listAttributes){
        Instances instances = null;
        try{
            Remove remove = new Remove();
            remove.setOptions(listAttributes);
            remove.setInputFormat(data);
            instances = Filter.useFilter(data, remove);
        } catch (Exception e){
            System.err.println("Error in removing attribute");
            System.err.println(e);   
        } finally {
            return instances;
        }
    }

    @SuppressWarnings("finally")
    public static Instances SMOTE(Instances data){
        Instances instances = null;
        try{
            SMOTE smote = new SMOTE();
            String[] opt = new String[]{"-P", "10.0"};
            smote.setOptions(opt);
            smote.setInputFormat(data); 
            instances = Filter.useFilter(data, smote);
            while (!checkEqual(instances)) {
                smote.setInputFormat(instances);
                instances = Filter.useFilter(instances, smote);
            }
        } catch (Exception e){
            System.err.println("Error in SMOTE!");
            System.err.println(e);
        } finally {
            return instances;
        }
    }



    private static boolean checkEqual(Instances data){
        AttributeStats attributeStats = data.attributeStats(0);
        int[] count = attributeStats.nominalCounts;
        int min = count[0];
        int max = count[0];
        for (int i=0; i<count.length; ++i){
            if (min>count[i]) min = count[i];
            if (max<count[i]) max = count[i];
        }
        if (((float) (max-min)/max) > 0.9) return true;
        else return false;
    }


}
