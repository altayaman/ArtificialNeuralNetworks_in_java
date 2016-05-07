
public class ANN_Test {

	public static void main(String[] args) {

		double[][] X = { {3.0/10, 5.0/5}, 
						 {5.0/10, 1.0/5}, 
						 {10.0/10, 2.0/5} };
		double[] y = {0.75, 0.82, 0.93}; 

		
		ANN NN = new ANN();
//		NN.print_2D_Matrix(NN.Weights1, "");
//		NN.print_1D_Matrix(NN.Weights2, "");
		NN.print_1D_Matrix(y, "Original y values");
//		double[][] W1 = NN.Weights1;
//		double[] W2 = NN.Weights2;
//		NN.print_2D_Matrix(W1);
//		NN.print_1D_Matrix(W2);
//		double[] yHat = NN.calculate_yHat_out(X);
//		NN.print_1D_Matrix(yHat);
		
		NN.gradientRunner(X, y, 0.05, 1000000);

//		W1 = NN.Weights1;
//		W2 = NN.Weights2;
//		NN.print_2D_Matrix(W1);
//		NN.print_1D_Matrix(W2);
		double[] yHat = NN.calculate_yHat_out(X);
		NN.print_1D_Matrix(yHat, "yHat values by generated weights");
		
//		double[] W2_deriv = NN.weights2_derivative(X, y);
//		NN.print_1D_Matrix(W2_deriv);
//		
//		double[][] W1_deriv = NN.weights1_derivative(X, y);
//		NN.print_2D_Matrix(W1_deriv);
	}

}
