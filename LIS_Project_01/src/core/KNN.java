package core;

import java.util.ArrayList;

/**
 * Linear scan algorithm which calculates the K-nearest neighbors to a given test point and returning some weighted average.
 * Training runs in O(1) per training point.
 * Testing runs in O(n*k) per testing point where n is the number of training samples and k some constant. This could probably be optimised to O(n+k)
 */

public class KNN extends NN_Algorithm {
	
	//Variables for feature normalisation.
	//These are necessary in order to be able to normalise values between [0,1] so distance metrics (specifically l2 norm) make more sense.
	//Normalising should lead to better results for many problems.
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
	
	/**
	 * Normalizes contained trainingSamples as well as future testing samples.
	 * TODO: after calling this future training samples should be normalized as well.
	 */
	public void normalize() {
		normalized = true;
		ArrayList<FeatureResultPair> trainingData = new ArrayList<FeatureResultPair>();
		normalisationMinimum = new double[trainingSamples.get(0).featureVec.length];
		normalisationMaximum = new double[normalisationMinimum.length];
		for (int j = 0; j < normalisationMinimum.length; j++) {
			normalisationMinimum[j] = trainingSamples.get(0).featureVec[j];
			normalisationMaximum[j] = trainingSamples.get(0).featureVec[j];
		}
		for (int i = 0; i < trainingSamples.size(); i++) {
			for (int j = 0; j < normalisationMinimum.length; j++) {
				if (normalisationMinimum[j] < trainingSamples.get(i).featureVec[j]) normalisationMinimum[j] = trainingSamples.get(i).featureVec[j];
				if (normalisationMaximum[j] > trainingSamples.get(i).featureVec[j]) normalisationMaximum[j] = trainingSamples.get(i).featureVec[j];
			}
		}
		for (int i = 0; i < trainingSamples.size(); i++) {
			trainingData.add(new FeatureResultPair(normaliseFeatures(trainingSamples.get(i).featureVec), trainingSamples.get(i).result));
		}
		trainingSamples = trainingData;
	}
	
	/**
	 * @return Number of closest neighbors algorithm uses at test time in order to calculate a return value.
	 */
	public int getK() {
		return K;
	}
	
	/**
	 * @param K : Number of closest neighbors algorithm uses at test time in order to calculate a return value.
	 */
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
		oldDistance = distance(closest.featureVec, features);
		for (int i = 1; i < trainingSamples.size(); i++) {
			newDistance = distance(trainingSamples.get(i).featureVec, features);
			if (newDistance < oldDistance) {
				oldDistance = newDistance;
				closest = trainingSamples.get(i);
			}
		}
		return (int) closest.result;
	}

	@Override
	public double regClassify(double[] features) {
		double oldDistance, newDistance;
		if (normalized) features = normaliseFeatures(features);
		FeatureResultPair closest = trainingSamples.get(0);
		oldDistance = distance(closest.featureVec, features);
		double distanceSum = 0, minDistance = -1;
		ArrayList<FeatureResultPair> closestK = new ArrayList<FeatureResultPair>();
		while (closestK.size() < K) {
			for (int i = 0; i < trainingSamples.size(); i++) {
				newDistance = distance(trainingSamples.get(i).featureVec, features);
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
		if (distance(closestK.get(0).featureVec, features) == 0) return closestK.get(0).result;
		double res = 0;
		for (FeatureResultPair sample : closestK) {
			res += sample.result * (1 /(distance(sample.featureVec, features))) / distanceSum;
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
