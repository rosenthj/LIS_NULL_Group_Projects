package core;

/**
 * Class for representing a training sample for a supervised learning algorithm.
 */

public class FeatureResultPair {
	public final double[] featureVec;
	public final double result;
	public FeatureResultPair(double[] x, double y) {this.featureVec = x; this.result = y;}
}
