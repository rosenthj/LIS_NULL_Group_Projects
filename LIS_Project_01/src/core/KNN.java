package core;

import java.util.ArrayList;

public class KNN extends NN_Algorithm {
	private double[] normalisationMinimum, normalisationMaximum;
	private boolean normalized = false;
	private int K;
	private ArrayList<FeatureResultPair> trainingSamples = new ArrayList<FeatureResultPair>();
	
	private double[] normaliseFeatures(double[] features) {
		double[] normFeatures = new double[features.length];
		for (int i = 0; i < features.length; i++) {
			normFeatures[i] = (features[i] - normalisationMinimum[i]) / (normalisationMaximum[i] - normalisationMinimum[i]);
		}
		return normFeatures;
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
			trainingData.add(new FeatureResultPair(normaliseFeatures(trainingSamples.get(i).x), trainingSamples.get(i).y));
		}
		trainingSamples = trainingData;
	}
	
	public int getK() {
		return K;
	}
	
	public void setK(int K) {
		this.K = K;
	}
	
	public void setTrainingSamples(ArrayList<FeatureResultPair> trainingSamples) {
		this.trainingSamples = trainingSamples;
	}
	
	@Override
	public int classify(double[] features) {
		if (normalized) features = normaliseFeatures(features);
		double oldDistance, newDistance;
		FeatureResultPair closest = trainingSamples.get(0);
		oldDistance = distance(closest.x, features);
		for (int i = 1; i < trainingSamples.size(); i++) {
			newDistance = distance(trainingSamples.get(i).x, features);
			if (newDistance < oldDistance) {
				oldDistance = newDistance;
				closest = trainingSamples.get(i);
			}
		}
		return (int) closest.y;
	}

	@Override
	public double regClassify(double[] features) {
		double oldDistance, newDistance;
		if (normalized) features = normaliseFeatures(features);
		FeatureResultPair closest = trainingSamples.get(0);
		oldDistance = distance(closest.x, features);
		double distanceSum = 0, minDistance = -1;
		ArrayList<FeatureResultPair> closestK = new ArrayList<FeatureResultPair>();
		while (closestK.size() < K) {
			for (int i = 0; i < trainingSamples.size(); i++) {
				newDistance = distance(trainingSamples.get(i).x, features);
				if (newDistance < oldDistance && newDistance > minDistance) {
					oldDistance = newDistance;
					closest = trainingSamples.get(i);
				}
			}
			distanceSum += 1 / oldDistance;
			minDistance = oldDistance;
			closestK.add(closest);
			oldDistance = Double.MAX_VALUE;
		}
		if (distance(closestK.get(0).x, features) == 0) return closestK.get(0).y;
		double res = 0;
		for (FeatureResultPair sample : closestK) {
			res += sample.y * (1 /(distance(sample.x, features))) / distanceSum;
		}
		return res;
	}

	@Override
	public void addTrainingSample(FeatureResultPair trainingSample) {
		trainingSamples.add(trainingSample);
	}

	@Override
	public void train(ArrayList<FeatureResultPair> trainingSamples) {
		// TODO Auto-generated method stub
		
	}
	
}
