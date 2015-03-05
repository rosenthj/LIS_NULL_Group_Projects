package core;

public class Polynomial implements Classifier {
	double[] weights;
	
	public Polynomial(int numVariables) {
		weights = new double[numVariables+1];//f(x0,x1,...,xn) = w0 + w1*x0 + w2*x1 +....
	}
	
	public double evaluateAt(double[] x) {
		assert weights.length == x.length+1 : "Invalid dimensions for function evaluation";
		double res = weights[0];
		for (int i = 0; i < x.length; i++) {
			res += weights[i+1] * x[i];
		}
		return res;
	}
	
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
