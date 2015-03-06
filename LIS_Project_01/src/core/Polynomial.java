package core;

/**
 * Class representing a polynomial function. It contains weights but not variables.
 * Instead variables must be provided separately in order to get the polynomials value.
 */

public class Polynomial implements Classifier {
	double[] weights;
	
	public Polynomial(int numVariables) {
		weights = new double[numVariables+1];//f(x0,x1,...,xn) = w0 + w1*x0 + w2*x1 +....
	}
	
	/**
	 * Evaluates the polynomial given x.
	 * @param x vector of values
	 * @return value of polynomial given variables x
	 */
	public double evaluateAt(double[] x) {
		assert weights.length == x.length+1 : "Invalid dimensions for function evaluation";
		double res = weights[0];//Polynomial is of the form a0 + a1x0 + a2x1 +...
		for (int i = 0; i < x.length; i++) {
			res += weights[i+1] * x[i];
		}
		return res;
	}
	
	/**
	 * Use this instead of evaluateAt(x) when an integral result is required.
	 */
	public long integerEvaluationAt(double[] x) {
		return Math.round(evaluateAt(x));
	}
	
	public void printWeights() {
		for (double weight : weights) System.out.print(weight + " ");
		System.out.println();
	}

	@Override
	public int classify(double[] features) {
		return (int) Math.round(evaluateAt(features));
	}

	@Override
	public double regClassify(double[] features) {
		return evaluateAt(features);
	}
	
}
