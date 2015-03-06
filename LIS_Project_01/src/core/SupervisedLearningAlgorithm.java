package core;

import java.util.ArrayList;
/**
 * Interface for Supervised learning algorithms.
 */
public interface SupervisedLearningAlgorithm {
	
	/**
	 * Add a single training sample to the algorithm.
	 * For online algorithms this may mean training immediately on the sample
	 * while for batch learning algorithms this should just add a sample to the list
	 */
	public void addTrainingSample(FeatureResultPair trainingSample);
	
	/**
	 * Traing algorithm on the param list of training samples and only on that list.
	 */
	public void train(ArrayList<FeatureResultPair> trainingSamples);
}
