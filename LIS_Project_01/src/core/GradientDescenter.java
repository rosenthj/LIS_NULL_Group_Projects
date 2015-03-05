package core;
import java.util.ArrayList;
import java.util.List;



public class GradientDescenter {
	
	private ArrayList<FeatureResultPair> trainingCases = new ArrayList<FeatureResultPair>();
	private Polynomial predictedFunction;
	
	public GradientDescenter(int dimensions) {
		//The weights for our initial guess will all be 0.
		predictedFunction = new Polynomial(dimensions);
	}
	
	public GradientDescenter(List<FeatureResultPair> trainingData) {
		for (FeatureResultPair trainingCase : trainingData) trainingCases.add(trainingCase);
		predictedFunction = new Polynomial(trainingCases.get(0).x.length);
	}
	
	public void addTrainingPoint(double[] x, double y) {
		trainingCases.add(new FeatureResultPair(x, y));
	}
	
	public double getAverageTrainingError() {
		double error = 0, caseError;
		for (FeatureResultPair trainingPoint : trainingCases) {
			caseError = trainingPoint.y - predictedFunction.evaluateAt(trainingPoint.x);
			error += caseError * caseError;
		}
		if (trainingCases.size() > 0) error /= trainingCases.size();
		return Math.sqrt(error);
	}
	
	public double[] getGradient() {
		double[] gradient = new double[predictedFunction.weights.length];
		for (FeatureResultPair trainingPoint : trainingCases) {
			double difference = trainingPoint.y - predictedFunction.evaluateAt(trainingPoint.x);
			gradient[0] += difference;//constant offset.
			for (int i = 0; i < trainingPoint.x.length; i++) {
				gradient[i+1] += difference * trainingPoint.x[i];
			}
		}
		for (int i = 0; i < gradient.length; i++) gradient[i] *= (2);
		return gradient;
	}
	
	/**
	 * Performs gradient descent on training examples until
	 * either the average training error is below the param threshold
	 * or the param maximum number of iterations has been evaluated.
	 * The learning rate is the size of each change relative to the gradient.
	 * @param tolerance
	 * @param maxIterations
	 * @param initialLearningRate
	 */
	public void learn(double tolerance, int maxIterations, double initialLearningRate) {
		int completedIterations = 0;
		System.out.println("learn(" + tolerance + "," + maxIterations + "," + initialLearningRate + ");");
		double error = getAverageTrainingError(), previousError = Double.MAX_VALUE;
		System.out.println("Average Training Error " + getAverageTrainingError());
		while (error > tolerance && completedIterations < maxIterations) {
			double[] gradient = getGradient();
			for (int i = 0; i < gradient.length; i++) predictedFunction.weights[i] += gradient[i] * initialLearningRate;
			completedIterations++;
			previousError = error;
			error = getAverageTrainingError();
			initialLearningRate = getNewLearningRate(previousError, error, initialLearningRate);
			if (completedIterations % Main.ITERATIONS_PER_TRAINING_ERROR_OUTPUT == 0) System.out.println("Average Training Error " + getAverageTrainingError());
		}
		if (completedIterations == maxIterations) System.out.println("Max iterations reached!");
	}
	
	private double getNewLearningRate(double previousError, double newError, double currentLearningRate) {
		if (previousError <= newError) return currentLearningRate / 2;
		return currentLearningRate * 1.1;
	}
	
	public Polynomial getPredictedFunction() {
		return predictedFunction;
	}
}
