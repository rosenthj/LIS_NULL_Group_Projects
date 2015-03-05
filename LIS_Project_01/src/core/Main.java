package core;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

public class Main {
	
	static final boolean DEBUG = true;
	static final int ITERATIONS_PER_TRAINING_ERROR_OUTPUT = 2500;
	static final int DIMENSION_TIME_OF_DAY = 4;
	static final int DIMENSIONS = 1;
	
	int[] monthsTable = {0,3,3,6,1,4,6,2,5,0,3,5};//Does not handle leap years.

	public static void main(String[] args) {
		Classifier p = getPredictor(1, 100000);
		makePredictions(p, "validate.csv", "validate_4NN_y.csv");
	}
	
	public static void makePredictions(Classifier p, String fileNameIn, String fileNameOut) {
		File inFile = new File(fileNameIn);
		File outFile = new File(fileNameOut);
		try {
			BufferedReader reader = new BufferedReader(new FileReader(inFile));
			BufferedWriter writer = new BufferedWriter(new FileWriter(outFile));
			String in = reader.readLine();
			while (in != null) {
				double[] x = getFeatureVec(in);
				double y = p.regClassify(x);
				writer.write(y + "\n");
				in = reader.readLine();
			}
			reader.close();
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
//	public static Classifier getPredictor(double tolerance, int maxIterations) {
//		ArrayList<FeatureResultPair> trainingData = new ArrayList<FeatureResultPair>();
//		try {
//			File file = new File("train.csv");
//			BufferedReader featureReader = new BufferedReader(new FileReader(file));
//			File resultFile = new File("train_y.csv");
//			BufferedReader resultReader = new BufferedReader(new FileReader(resultFile));
//			String featureLine = featureReader.readLine();
//			String result = resultReader.readLine();
//			while (featureLine != null) {
////				System.out.println(featureLine);
//				double[] x = getFeatureVec(featureLine);
//				double y = Double.parseDouble(result);
//				trainingData.add(new FeatureResultPair(x, y));
//				featureLine = featureReader.readLine();
//				result = resultReader.readLine();
//			}
//			System.out.println("Successfully added " + trainingData.size() + " training data points");
//			featureReader.close();
//			resultReader.close();
//		} catch (IOException e) {
//			e.printStackTrace();
//		}
//		GradientDescenter predictor = new GradientDescenter(trainingData);
//		predictor.learn(tolerance, maxIterations, 0.00000000000001 / (Math.pow(10, DIMENSIONS)));
//		System.out.println("Final training error: " + predictor.getAverageTrainingError());
//		predictor.getPredictedFunction().printWeights();
//		return predictor.getPredictedFunction();
//	}
	
	public static Classifier getPredictor(double tolerance, int maxIterations) {
		ArrayList<FeatureResultPair> trainingData = new ArrayList<FeatureResultPair>();
		try {
			File file = new File("train.csv");
			BufferedReader featureReader = new BufferedReader(new FileReader(file));
			File resultFile = new File("train_y.csv");
			BufferedReader resultReader = new BufferedReader(new FileReader(resultFile));
			String featureLine = featureReader.readLine();
			String result = resultReader.readLine();
			while (featureLine != null) {
//				System.out.println(featureLine);
				double[] x = getFeatureVec(featureLine);
				double y = Double.parseDouble(result);
				trainingData.add(new FeatureResultPair(x, y));
				featureLine = featureReader.readLine();
				result = resultReader.readLine();
			}
			System.out.println("Successfully added " + trainingData.size() + " training data points");
			featureReader.close();
			resultReader.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		KNN knn = new KNN();
		knn.setTrainingSamples(trainingData);
		knn.normalize();
		knn.setK(4);
		return knn;
	}
	
	public static double[] getFeatureVec(String featureLine) {
		String[] tokens = featureLine.split(",");
		double[] x = new double[tokens.length + 1 + DIMENSION_TIME_OF_DAY];
		String[] date = tokens[0].split(" ");
		int dayOfMonth = Integer.parseInt(date[0].split("-")[2]);
		int month = Integer.parseInt(date[0].split("-")[1]);
		int year = Integer.parseInt(date[0].split("-")[0]) % 100;
		int weekDay = dayOfMonth + month + year + year/4 + 6;
		x[0] = (year - 13) * 12 + month;
		x[0] /= 12;
		x[1] = weekDay;
		x[1] /= 6;
		String[] timeTokens = date[1].split(":");
		x[2] = (Double.parseDouble(timeTokens[0]) + Double.parseDouble(timeTokens[1])/60) / 24;
		for (int i = 1; i < DIMENSION_TIME_OF_DAY; i++) {
			x[2+i] = Math.pow(x[2], i+1);
		}
		for (int i = 2 + DIMENSION_TIME_OF_DAY; i < x.length; i++) x[i] = Double.parseDouble(tokens[i-1-DIMENSION_TIME_OF_DAY]);
		double[] highDimX = new double[x.length * DIMENSIONS];
		for (int i = 0; i < x.length; i++) {
			highDimX[DIMENSIONS*i] = x[i];
			for (int d = 1; d < DIMENSIONS; d++) {
				highDimX[DIMENSIONS*i + d] = highDimX[DIMENSIONS*i + d-1] * x[i];
			}
		}
		return highDimX;
	}
	
}
