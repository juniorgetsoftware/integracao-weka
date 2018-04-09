package br.com.ppgcc;

import weka.classifiers.bayes.NaiveBayes;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class ExemploRiscoDeVenda {

	public static void main(String[] args) {
		try {
			DataSource ds = new DataSource("vendas.arff");
			Instances ins = ds.getDataSet();
			
			ins.setClassIndex(3);
			
			NaiveBayes nb = new NaiveBayes();
			nb.buildClassifier(ins);
			
			Instance novo = new DenseInstance(4);
			novo.setDataset(ins);
			
			novo.setValue(0, "M");
			novo.setValue(1, "20-39");
			novo.setValue(2, "Nao");
			
			double probabilidade[] = nb.distributionForInstance(novo);
			System.out.println("sim "+probabilidade[1]*100);
			System.out.println("não "+probabilidade[0]*100);
			
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

}
