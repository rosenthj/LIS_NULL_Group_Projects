package core;

public interface Classifier {
	
	/**
	 * Predict result of classification problem
	 * @param features
	 * @return integer representing predicted class
	 */
	public int classify(double[] features);
	
	/**
	 * Predict result of regression problem
	 * @param features
	 * @return real value result
	 */
	public double regClassify(double[] features);
}
