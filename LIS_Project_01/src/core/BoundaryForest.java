package core;

import java.util.ArrayList;

public class BoundaryForest implements Classifier, SupervisedLearningAlgorithm {
	
	private int threshold = 1;
	private double[] normalisationMinimum, normalisationMaximum;
	private boolean normalized = false;
	private ArrayList<FeatureResultPair> trainingSamples = new ArrayList<FeatureResultPair>();
	
	private ArrayList<BoundaryTree> trees = new ArrayList<BoundaryTree>();
	
	private double[] normaliseFeatures(double[] features) {
		double[] normFeatures = new double[features.length];
		for (int i = 0; i < features.length; i++) {
			normFeatures[i] = (features[i] - normalisationMinimum[i]) / (normalisationMaximum[i] - normalisationMinimum[i]);
		}
		return normFeatures;
	}
	
	private double normaliseResults(double y) {
		if (threshold != 1) {
			long ny = Math.round(y);
			ny /= threshold;
			ny *= threshold;
			return ny + threshold/2;
		}
		return y;
	}
	
	public void setThreshold(int threshold) {
		this.threshold = threshold;
	}
	
	public void normalize() {
		normalized = true;
		ArrayList<FeatureResultPair> trainingData = new ArrayList<FeatureResultPair>();
		normalisationMinimum = new double[trainingSamples.get(0).x.length];
		normalisationMaximum = new double[normalisationMinimum.length];
		for (int j = 0; j < normalisationMinimum.length; j++) {
			normalisationMinimum[j] = trainingSamples.get(0).x[j];
			normalisationMaximum[j] = trainingSamples.get(0).x[j];
		}
		for (int i = 0; i < trainingSamples.size(); i++) {
			for (int j = 0; j < normalisationMinimum.length; j++) {
				if (normalisationMinimum[j] < trainingSamples.get(i).x[j]) normalisationMinimum[j] = trainingSamples.get(i).x[j];
				if (normalisationMaximum[j] > trainingSamples.get(i).x[j]) normalisationMaximum[j] = trainingSamples.get(i).x[j];
			}
		}
		for (int i = 0; i < trainingSamples.size(); i++) {
			trainingData.add(new FeatureResultPair(normaliseFeatures(trainingSamples.get(i).x), normaliseResults(trainingSamples.get(i).y)));
		}
		trainingSamples = trainingData;
	}
	
	public void setNumTrees(int numTrees) {
		trees.clear();
		for (int i = 0; i < numTrees; i++) trees.add(new BoundaryTree());
	}
	
	public int getNumTrees() {
		return trees.size();
	}
	
	@Override
	public void train(ArrayList<FeatureResultPair> trainingSamples) {
		this.trainingSamples = trainingSamples;
		normalize();
		for (int i = 0; i < trees.size(); i++) {
			int s = i * this.trainingSamples.size() / trees.size();
			for (int a = 0; a < 2; a++) {
				for (int j = s; j < this.trainingSamples.size(); j++) trees.get(i).addTrainingSample(this.trainingSamples.get(j));
				for (int j = 0; j < s; j++) trees.get(i).addTrainingSample(this.trainingSamples.get(j));
			}
		}
	}
	
	@Override
	public int classify(double[] features) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public double regClassify(double[] features) {
		double res = 0;
		if (normalized) features = normaliseFeatures(features);
		for (BoundaryTree tree : trees) res += tree.regClassify(features);
		return res / trees.size();
	}

	@Override
	public void addTrainingSample(FeatureResultPair trainingSample) {
		// TODO Auto-generated method stub
		
	}

}
