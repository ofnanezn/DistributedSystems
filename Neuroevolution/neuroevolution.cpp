#include <iostream>
#include <vector>
#include <math.h>
#include <random>

using namespace std;

float* linear_forward(float*, float*);
float* sigmoid(float*);
float* relu(float*);

class NeuralNetwork{
	int num_layers;
	int input_size;
	int output_size;
	int* hidden_layers;
	string* activations;
	int training_samples;

	public:
		neural_network(int num_layers, int input_size, int output_size, int* hidden_layers, string* activations, int training_samples){
			num_layers = num_layers;
			hidden_layers = hidden_layers;
			activations = activations;
			input_size = input_size;
			output_size = output_size;
			training_samples = training_samples;
		}

		vector<float*> initialize_weights();
		float* forward_propagate(float*, float*	);
		float cost(float*, float*)
};	

vector<float*> NeuralNetwork::initialize_weights(){
	default_random_engine generator;
  	normal_distribution<double> distribution(0.0,1.0);

	vector<float*> W;
	
	float* W0 = new float[input_size * hidden_layers[0]];
	for(int i = 0; i < input_size * hidden_layers[0]; ++i){
		double r = distribution(generator);
		double bound = sqrt(1.0 / (input_size + hidden_layers[0]));
		W0[i] = r * bound;
	}
	W.push_back(W0);

	for(int l = 1; l < num_layers; ++l){
		float* Wl = new float[hidden_layers[l-1] * hidden_layers[l]];
		for(int i = 0; i < hidden_layers[l-1] * hidden_layers[l]; ++i){
			double r = distribution(generator);
			double bound = sqrt(1.0 / (hidden_layers[l] + hidden_layers[l-1]));
			Wl[i] = r * bound;
		}
		W.push_back(Wl)
	}

	float* WL = new float[output_size * hidden_layers[num_layers-1]];
	for(int i = 0; i < output_size * hidden_layers[num_layers-1]; ++i){
		double r = distribution(generator);
		double bound = sqrt(1.0 / (output_size + hidden_layers[num_layers-1]));
		WL[i] = r * bound;
	}
	W.push_back(WL);

	return W;
}

float* NeuralNetwork::forward_propagate(float* input_layer, float* W){
	float* A_prev, A, Z;
	int current_size = training_samples * hidden_layers[0];
	Z = linear_forward(input_layer, W[0], training_samples, input_size, hidden_layers[0]);
	if(activations[0] == "sigmoid")
		A = sigmoid(Z, current_size);

	for (int l = 1; l < num_layers; ++l){
		A_prev = A;
		
	}
}

int main(){
	int num_layers = 4;
}


float* linear_forward(float* A_prev, float* W, int m, int p, int n){
	float* Z = new float[m * n];
	for(int i = 0; i < m; i++){
		for (int j = 0; j < n; j++){
			Z[i*(p + 1) + j] = 0;
			for (int k = 0; k < p; ++k)
				Z[i*(p + 1) + j] += A_prev[i * p + k] * W[j + k * (p + 1)];
		}
	}
	return Z;
}

float* sigmoid(float* Z, int size){
	float* A = new float[size];
	for(int i = 0; i < size; ++i)
		A[i] = 1.0 / (1 + exp(-Z[i]));
	return A;
}