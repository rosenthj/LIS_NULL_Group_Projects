package core;

/**
 * Super class for all nearest neighbor algorithms. 
 * Nearest neighbor algorithms rely on some distance metric which is defined here.
 */

public abstract class NN_Algorithm implements SupervisedLearningAlgorithm {
	
	public static double distance(double[] x1, double[] x2) {
		double res = 0, dif;
		for (int i = 0; i < x1.length; i++) {
			dif = x1[i] - x2[i];
			res += dif * dif;
		}
		return Math.sqrt(res);
	}
}
