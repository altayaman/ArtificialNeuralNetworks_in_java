import java.text.DecimalFormat;
import java.util.Random;


public class ANN {

	public int inputLayer = 2;
	public int hiddenLayer = 3;
	public int outputLayer = 1;
	
	double[][] Weights1;
	double[] Weights2;

	/*  Weights1    *  Weights2
	 *              *
	 *  Wa1  Wa2    *  Wd1
	 *  Wb1  Wb2    *  Wd2
	 *  Wc1  Wc2    *  Wd3
	 *              * 
	 */
	
	/*  X1   X2
	 *  a1   a2
	 *  b1   b2
	 *  c1   c2
	 *  
	 */

	
	public ANN(){
		Weights1 = new double[hiddenLayer][inputLayer];
		Weights2 = new double[hiddenLayer];
		
		Random rand = new Random();
		
//		rand.setSeed(2);
		for (int i = 0; i < hiddenLayer; i++) {
			for (int j = 0; j < inputLayer; j++) {
			Weights1[i][j] = Math.round(rand.nextDouble() * 100.0) / 100.0;
			Weights2[i] = Math.round(rand.nextDouble() * 100.0) / 100.0;
		}}
		
//		Weights1[0][0] = 0.73;   Weights1[0][1] = 0.5;
//		Weights1[1][0] = 0.86;   Weights1[1][1] = 0.23;
//		Weights1[2][0] = 0.74;   Weights1[2][1] = 0.03;
//		
//		Weights2[0] = 0.99;
//		Weights2[1] = 0.07;
//		Weights2[2] = 0.69;
	}
	
	/*
	 *	Sigmoid function
	 */
	public double sigmoid(double d){
		return 1/(1 + Math.exp(-d));
	}
	
	/*
	 *	SigmoidPrime function
	 */
	public double sigmoidPrime(double d){
		return Math.exp(-d)/Math.pow((1 + Math.exp(-d)), 2);
	}
	
	/*
	 * Layer1_in calculation
	 */
	public double[][] calculate_Layer1_in(double[][] X){
		
		double[][] Layer1_in = new double[hiddenLayer][X.length];
		
		for (int Xi = 0; Xi < X.length; Xi++) {
			for (int hidden = 0; hidden < hiddenLayer; hidden++) {
				Layer1_in[hidden][Xi] = 0.0;
		}}
		
		
		for (int Xi = 0; Xi < X.length; Xi++) {
			for (int hidden = 0; hidden < hiddenLayer; hidden++) {
				for (int input = 0; input < inputLayer; input++) {
					Layer1_in[hidden][Xi] = Layer1_in[hidden][Xi] + Weights1[hidden][input] * X[Xi][input];
//					System.out.print("Weights1[" + hidden + "][" + input + "] = " + Weights1[hidden][input] + "   ");
//					System.out.println("X[" + Xi + "][" + input + "] = " + X[Xi][input] );
//					System.out.println("Weights1[" + hidden + "][" + input + "] * X[" + Xi + "][" + input + "] = " + Weights1[hidden][input] * X[Xi][input]);
		}}}

		return Layer1_in;
	}
	
	/*
	 * Layer1_out calculation
	 */
	public double[][] calculate_Layer1_out(double[][] X){
		
		double[][] Layer1_out = calculate_Layer1_in(X); 
		
		for (int i = 0; i < Layer1_out.length; i++) {
			for (int j = 0; j < Layer1_out[0].length; j++) {
				Layer1_out[i][j] = sigmoid(Layer1_out[i][j]);
		}}

		return Layer1_out;
	}
	
	/*
	 * yHat_in calculation
	 */
	public double[] calculate_yHat_in(double[][] X){
		double[][] Layer1_out = calculate_Layer1_out(X);
		double[] yHat_in = new double[X.length];
		
		for (int i = 0; i < yHat_in.length; i++) {
			yHat_in[i] = 0.0;
		}
		
		for (int j = 0; j < Layer1_out[0].length; j++) {
			for (int i = 0; i < Layer1_out.length; i++) {
				yHat_in[j] = yHat_in[j] + Weights2[i] * Layer1_out[i][j];
		}}
		
		return yHat_in;
	}
	
	/*
	 * yHat_out calculation
	 */
	public double[] calculate_yHat_out(double[][] X){
		double[] yHat_out = calculate_yHat_in(X);
		
		for (int i = 0; i < yHat_out.length; i++) {
			yHat_out[i] = sigmoid(yHat_out[i]);
		}
		
		return yHat_out;
	}
	
	/*
	 *	W2 derivation
	 */
	public double[] weights2_derivative(double[][] X, double[] y){
		double[] yHat_in = calculate_yHat_in(X);
		double[] yHat_out = calculate_yHat_out(X);
		double[][] L1_out = calculate_Layer1_out(X);
		
		double Wd1_prime = -(y[0] - yHat_out[0]) * sigmoidPrime(yHat_in[0]) * L1_out[0][0]
						   -(y[1] - yHat_out[1]) * sigmoidPrime(yHat_in[1]) * L1_out[0][1]
						   -(y[2] - yHat_out[2]) * sigmoidPrime(yHat_in[2]) * L1_out[0][2];
		
		double Wd2_prime = -(y[0] - yHat_out[0]) * sigmoidPrime(yHat_in[0]) * L1_out[1][0]
				  		   -(y[1] - yHat_out[1]) * sigmoidPrime(yHat_in[1]) * L1_out[1][1]
				  		   -(y[2] - yHat_out[2]) * sigmoidPrime(yHat_in[2]) * L1_out[1][2];
		
		double Wd3_prime = -(y[0] - yHat_out[0]) * sigmoidPrime(yHat_in[0]) * L1_out[2][0]
						   -(y[1] - yHat_out[1]) * sigmoidPrime(yHat_in[1]) * L1_out[2][1]
						   -(y[2] - yHat_out[2]) * sigmoidPrime(yHat_in[2]) * L1_out[2][2];
		
		
		double[] W2_deriv = new double[Weights2.length];
		W2_deriv[0] = Wd1_prime;
		W2_deriv[1] = Wd2_prime;
		W2_deriv[2] = Wd3_prime;
		
		return W2_deriv;
	}
	
	
	/*
	 *	W1 derivation
	 */
	public double[][] weights1_derivative(double[][] X, double[] y){
		double[] yHat_in = calculate_yHat_in(X);
		double[] yHat_out = calculate_yHat_out(X);
		double[][] L1_out = calculate_Layer1_out(X);
		double[][] L1_in = calculate_Layer1_in(X);
		
		double Wa1_prime = -(y[0] - yHat_out[0]) * sigmoidPrime(yHat_in[0]) * Weights2[0] * sigmoidPrime(L1_in[0][0]) * X[0][0] +
						   -(y[1] - yHat_out[1]) * sigmoidPrime(yHat_in[1]) * Weights2[0] * sigmoidPrime(L1_in[0][1]) * X[1][0] +
						   -(y[2] - yHat_out[2]) * sigmoidPrime(yHat_in[2]) * Weights2[0] * sigmoidPrime(L1_in[0][2]) * X[2][0];
		
		double Wa2_prime = -(y[0] - yHat_out[0]) * sigmoidPrime(yHat_in[0]) * Weights2[0] * sigmoidPrime(L1_in[0][0]) * X[0][1] +
				  		   -(y[1] - yHat_out[1]) * sigmoidPrime(yHat_in[1]) * Weights2[0] * sigmoidPrime(L1_in[0][1]) * X[1][1] +
				  		   -(y[2] - yHat_out[2]) * sigmoidPrime(yHat_in[2]) * Weights2[0] * sigmoidPrime(L1_in[0][2]) * X[2][1];
		
		double Wb1_prime = -(y[0] - yHat_out[0]) * sigmoidPrime(yHat_in[0]) * Weights2[1] * sigmoidPrime(L1_in[1][0]) * X[0][0] +
						   -(y[1] - yHat_out[1]) * sigmoidPrime(yHat_in[1]) * Weights2[1] * sigmoidPrime(L1_in[1][1]) * X[1][0] +
						   -(y[2] - yHat_out[2]) * sigmoidPrime(yHat_in[2]) * Weights2[1] * sigmoidPrime(L1_in[1][2]) * X[2][0];
		
		double Wb2_prime = -(y[0] - yHat_out[0]) * sigmoidPrime(yHat_in[0]) * Weights2[1] * sigmoidPrime(L1_in[1][0]) * X[0][1] +
						   -(y[1] - yHat_out[1]) * sigmoidPrime(yHat_in[1]) * Weights2[1] * sigmoidPrime(L1_in[1][1]) * X[1][1] +
						   -(y[2] - yHat_out[2]) * sigmoidPrime(yHat_in[2]) * Weights2[1] * sigmoidPrime(L1_in[1][2]) * X[2][1];
		
		double Wc1_prime = -(y[0] - yHat_out[0]) * sigmoidPrime(yHat_in[0]) * Weights2[2] * sigmoidPrime(L1_in[2][0]) * X[0][0] +
				   		   -(y[1] - yHat_out[1]) * sigmoidPrime(yHat_in[1]) * Weights2[2] * sigmoidPrime(L1_in[2][1]) * X[1][0] +
				   		   -(y[2] - yHat_out[2]) * sigmoidPrime(yHat_in[2]) * Weights2[2] * sigmoidPrime(L1_in[2][2]) * X[2][0];
		
		double Wc2_prime = -(y[0] - yHat_out[0]) * sigmoidPrime(yHat_in[0]) * Weights2[2] * sigmoidPrime(L1_in[2][0]) * X[0][1] +
				   		   -(y[1] - yHat_out[1]) * sigmoidPrime(yHat_in[1]) * Weights2[2] * sigmoidPrime(L1_in[2][1]) * X[1][1] +
				   		   -(y[2] - yHat_out[2]) * sigmoidPrime(yHat_in[2]) * Weights2[2] * sigmoidPrime(L1_in[2][2]) * X[2][1];
		
		
		double[][] W1_deriv = new double[Weights1.length][Weights1[0].length];
		W1_deriv[0][0] = Wa1_prime;
		W1_deriv[0][1] = Wa2_prime;
		W1_deriv[1][0] = Wb1_prime;
		W1_deriv[1][1] = Wb2_prime;
		W1_deriv[2][0] = Wc1_prime;
		W1_deriv[2][1] = Wc2_prime;
		
		return W1_deriv;
	}
	
	public void gradientRunner(double[][] X, double[] y, double learningRate, int numOfIterations){
		double[][] W1_gradients = weights1_derivative(X, y);
		double[][] New_W1 = Weights1;
		
		double[] W2_gradients = weights2_derivative(X, y);
		double[] New_W2 = Weights2;
		
		
		for (int iter = 0; iter < numOfIterations; iter++) {
			W1_gradients = weights1_derivative(X, y);
			W2_gradients = weights2_derivative(X, y);
			
			for (int i = 0; i < Weights1.length; i++) {
				for (int j = 0; j < Weights1[0].length; j++) {
					Weights1[i][j] = Weights1[i][j] - W1_gradients[i][j] * learningRate;
			}}
			
			for (int i = 0; i < Weights1.length; i++) {
				Weights2[i] = Weights2[i] - W2_gradients[i] * learningRate;
			}
		
		}
		
	}
	
	/*
	 *	1st Weight values printing function
	 */
	void printWeights1(){
		System.out.println("Initialized weight values 1:\n");
		
		for (int i = 0; i < hiddenLayer; i++) {
			for (int j = 0; j < inputLayer; j++) {
				System.out.print(Weights1[i][j] + "  ");
			}
			System.out.println();
		}
		
		System.out.println();
	}
	
	
	/*
	 *	2nd Weight values printing function
	 */
	void printWeights2(){
		System.out.println("Initialized weight values2:\n");
		
		for (int i = 0; i < hiddenLayer; i++) {
				System.out.print(Weights2[i] + "  ");
			System.out.println();
		}
		
		System.out.println();
	}
	
	
	/*
	 *	2D matrix values printing function
	 */
	void print_2D_Matrix(double[][] matrix, String s){
//		System.out.println("2D Matrix values:\n");
		System.out.println(s + "\n");
		
		for (int i = 0; i < matrix.length; i++) {
			for (int j = 0; j < matrix[0].length; j++) {
				System.out.print(matrix[i][j] + "  ");
			}
			System.out.println();
		}
		
		System.out.println();
	}

	
	/*
	 *	1D matrix values printing function
	 */
	void print_1D_Matrix(double[] matrix, String s){
//		System.out.println("1D Matrix values:\n");
		System.out.println(s + "\n");
		
		for (int i = 0; i < matrix.length; i++) {
			System.out.print(matrix[i] + "  \n");
		}
		
		System.out.println();
	}
	
}
