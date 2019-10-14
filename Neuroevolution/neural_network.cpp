#include <iostream>
#include <vector>
#include <math.h>
#include <random>

using namespace std;

double* linear_forward(double*, vector<double*>, int, int, int);
double* sigmoid(double*);
double* relu(double*);

class NeuralNetwork{
	int num_layers;
	int input_size;
	int output_size;
	int* hidden_layers;
	string* activations;
	int training_samples;

	public:
		NeuralNetwork(int num_layers, int input_size, int output_size, int* hidden_layers, string* activations, int training_samples){
			num_layers = num_layers;
			hidden_layers = hidden_layers;
			activations = activations;
			input_size = input_size;
			output_size = output_size;
			training_samples = training_samples;
		}

		vector<double*> initialize_weights();
		double* forward_propagate(double*, vector<double*>);
		//double cost(double*, double*);
};	

vector<double*> NeuralNetwork::initialize_weights(){
	default_random_engine generator;
  	normal_distribution<double> distribution(0.0,1.0);

	vector<double*> W;
	
	double* W0 = new double[input_size * hidden_layers[0]];
	for(int i = 0; i < input_size * hidden_layers[0]; ++i){
		double r = distribution(generator);
		double bound = sqrt(1.0 / (input_size + hidden_layers[0]));
		W0[i] = r * bound;
	}
	W.push_back(W0);

	for(int l = 1; l < num_layers; ++l){
		double* Wl = new double[hidden_layers[l-1] * hidden_layers[l]];
		for(int i = 0; i < hidden_layers[l-1] * hidden_layers[l]; ++i){
			double r = distribution(generator);
			double bound = sqrt(1.0 / (hidden_layers[l] + hidden_layers[l-1]));
			Wl[i] = r * bound;
		}
		W.push_back(Wl);
	}

	double* WL = new double[output_size * hidden_layers[num_layers-1]];
	for(int i = 0; i < output_size * hidden_layers[num_layers-1]; ++i){
		double r = distribution(generator);
		double bound = sqrt(1.0 / (output_size + hidden_layers[num_layers-1]));
		WL[i] = r * bound;
	}
	W.push_back(WL);

	return W;
}

double* NeuralNetwork::forward_propagate(double* input_layer, vector<double*> W){
	double* A_prev, A, Z;
	int current_size = training_samples * hidden_layers[0];
	Z = linear_forward(input_layer, W[0], training_samples, input_size, hidden_layers[0]);
	if(activations[0] == "sigmoid")
		A = sigmoid(Z, current_size);

	for (int l = 1; l < num_layers; ++l){
		A_prev = A;
		current_size = training_samples * hidden_layers[l];
		Z = linear_forward(A_prev, W[l], training_samples, hidden_layers[l-1], hidden_layers[l]);
		if(activations[0] == "sigmoid")
			A = sigmoid(Z, current_size);
	}

	A_prev = A;
	current_size = training_samples * output_size;
	Z = linear_forward(A_prev, W[l], training_samples, hidden_layers[num_layers-1], output_size);
	if(activations[0] == "sigmoid")
		A = sigmoid(Z, current_size);
	return A;
}

double NeuralNetwork::cost(double* AL, double* Y){
	return 0;
}

int main(){
	int num_layers = 1, input_size = 2, output_size = 1, training_samples = 4;
	int *hidden_layers = new int[1];
	hidden_layers[0] = 2;
	string *activations = new string[2];
	activations[0] = "sigmoid";
	activations[1] = "sigmoid";

	vector<double*> W;

	NeuralNetwork nn(num_layers, input_size, output_size, hidden_layers, activations, training_samples);
	W = nn.initialize_weights();
	for(int i = 0; i < input_size * hidden_layers[0]; ++i){
		cout << W[i] << endl;
	}
}


double* linear_forward(double* A_prev, double* W, int m, int p, int n){
	double* Z = new double[m * n];
	for(int i = 0; i < m; i++){
		for (int j = 0; j < n; j++){
			Z[i*(p + 1) + j] = 0;
			for (int k = 0; k < p; ++k)
				Z[i*(p + 1) + j] += A_prev[i * p + k] * W[j + k * (p + 1)];
		}
	}
	return Z;
}

double* sigmoid(double* Z, int size){
	double* A = new double[size];
	for(int i = 0; i < size; ++i)
		A[i] = 1.0 / (1 + exp(-Z[i]));
	return A;
}
