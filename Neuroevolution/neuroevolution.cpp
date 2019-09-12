#include <iostream>
#include <string>

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
		float* forward_propagate(float*, float*, float);
		float cost(float*, float*)
};

float* NeuralNetwork::initialize_weights(int input_size, int output_size){
	vector<vector<vector<float>>> W;

}

int main(){
	int num_layers = 4;
}	