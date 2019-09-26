#include <iostream>
#include <string>
#include <individual.h>
#include <neural_network.h>

using namespace std;

class GeneticAlgorithm{
	int pop_size

	public:
		GeneticAlgorithm(int pop_size){
			pop_size = pop_size;
		}

		Individual initial_individual(int, int, int, int*, string*, int);
		vector<Individual> initialize_pop(int, int, int*, string*);
		Individual mutation(Individual);
		vector<Individual> run(int, double*, double*, int, int*, string*);
};

Individual GeneticAlgorithm::initial_individual(){
	double* W
	NeuralNetwork individual = new NeuralNetwork(int num_layers, int input_size, int output_size, 
											int* hidden_layers, string* activations, int training_samples);
	W = individual.initialize_weights();
	return new Individual(W, 0, activations);
}

vector<Individual> GeneticAlgorithm::initialize_pop(){

}
