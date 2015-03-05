package core;

import java.util.ArrayList;

public interface SupervisedLearningAlgorithm {
	public void addTrainingSample(FeatureResultPair trainingSample);
	public void train(ArrayList<FeatureResultPair> trainingSamples);
}
