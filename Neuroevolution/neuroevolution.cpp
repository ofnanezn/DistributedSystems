#include <iostream>
#include <vector>
#include <math.h>
#include <random>

using namespace std;

class NeuralNetwork{
	int num_layers;
	int* hidden_layers;
	string* activations;

	public:
		neural_network(int num_layers, int* hidden_layers, string* activations){
			num_layers = num_layers;
			hidden_layers = hidden_layers;
			activations = activations;
		}

		vector<float*> initialize_weights(int, int);
		float* forward_propagate(float*, float*	);
		float cost(float*, float*)
};	

vector<float*> NeuralNetwork::initialize_weights(int input_size, int output_size){
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

float* forward_propagate(float*, float*, int);
float* linear_forward(float*, float*);

int main(){
	int num_layers = 4;
}

float* forward_propagate(float* input_layer, float* W, int num_layers){
	float* A_prev, Z;
	float* A = input_layer;
	for (int l = 0; l < num_layers; ++l){
		A_prev = A;
		Z = linear_forward(A_prev, W[l]);
	}
}

