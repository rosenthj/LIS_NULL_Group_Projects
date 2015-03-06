package core;

import java.util.ArrayList;

public class BoundaryTree extends NN_Algorithm implements Classifier {
	
	private Node root = null;
	
	private class Node {
		ArrayList<Node> children = new ArrayList<Node>();
		final double[] x;
		final double y;
		public Node(FeatureResultPair values) {
			x = values.featureVec;
			y = values.result;
		}
		
		public void addChildWithFeatures(FeatureResultPair values) {
			children.add(new Node(values));
		}
	}
	
	public Node getNearestNode(Node currentNode, double distance, double[] features) {
		ArrayList<Node> candidateNodes = currentNode.children;
		double candidateDistance;
		Node closestNode = currentNode;
		for (Node node : candidateNodes) {
			candidateDistance = distance(node.x, features);
			if (candidateDistance < distance) {
				distance = candidateDistance;
				closestNode = node;
			}
		}
		if (closestNode == currentNode) return currentNode;
		return getNearestNode(closestNode, distance, features);
	}
	
	
	@Override
	public int classify(double[] features) {
		return (int) getNearestNode(root, distance(root.x, features), features).y;
	}

	@Override
	public double regClassify(double[] features) {
		return getNearestNode(root, distance(root.x, features), features).y;
	}

	@Override
	public void addTrainingSample(FeatureResultPair trainingSample) {
		if (root == null) {
			root = new Node(trainingSample);
		}
		else {
			Node node = getNearestNode(root, distance(root.x, trainingSample.featureVec), trainingSample.featureVec);
			if (node.y != trainingSample.result) node.addChildWithFeatures(trainingSample);
		}
	}


	@Override
	public void train(ArrayList<FeatureResultPair> trainingSamples) {
		for(FeatureResultPair sample : trainingSamples) addTrainingSample(sample);
	}

}
