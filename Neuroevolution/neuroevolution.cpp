#include <iostream>
#include <string>
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

		float* initialize_weights(int, int);
		float* forward_propagate(float*, float*);
		float cost(float*, float*)
};	

vector<vector<float> NeuralNetwork::initialize_weights(int input_size, int output_size){
	default_random_engine generator;
  	normal_distribution<double> distribution(0.0,1.0);

	vector<vector<float>> W;
	
	vector<float> W0(input_size * hidden_layers[0]);
	for(int i = 0; i < W0.size(); ++i){
		double r = distribution(generator);
		double bound = sqrt(1.0 / (input_size + hidden_layers[0]));
		W0[i] = r * bound;
	}
	W.push_back(W0);

	for(int l = 1; l < num_layers; ++l){
		vector<float> Wl(hidden_layers[l-1] * hidden_layers[l]);
		for(int i = 0; i < Wl.size(); ++i){
			double r = distribution(generator);
			double bound = sqrt(1.0 / (hidden_layers[l] + hidden_layers[l-1]));
			Wl[i] = r * bound;
		}
		W.push_back(Wl)
	}

	vector<float> WL(output_size * hidden_layers[num_layers-1]);
	for(int i = 0; i < WL.size(); ++i){
		double r = distribution(generator);
		double bound = sqrt(1.0 / (output_size + hidden_layers[num_layers-1]));
		WL[i] = r * bound;
	}
	W.push_back(WL);

	return W;
}

vector<float> forward_propagate();

int main(){
	int num_layers = 4;
}	