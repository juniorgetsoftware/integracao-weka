package br.com.ppgcc;

import weka.classifiers.bayes.NaiveBayes;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class ExemploDiabetes {

	public static void main(String[] args) {
		try {
			DataSource ds = new DataSource("diabetes.arff");
			Instances ins = ds.getDataSet();
//			System.out.println(ins.toString());
			ins.setClassIndex(8);
			
			NaiveBayes nb = new NaiveBayes();
			nb.buildClassifier(ins);
			
			Instance novo = new DenseInstance(9);
			novo.setDataset(ins);
			
			novo.setValue(0, 0);
			novo.setValue(1, 137);
			novo.setValue(2, 40);
			novo.setValue(3, 35);
			novo.setValue(4, 168);
			novo.setValue(5, 43.1);
			novo.setValue(6, 2.288);
			novo.setValue(7, 27);
			
			double probabilidade[] = nb.distributionForInstance(novo);
			System.out.println("sim "+probabilidade[1]*100);
			System.out.println("não "+probabilidade[0]*100);
			
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

}
