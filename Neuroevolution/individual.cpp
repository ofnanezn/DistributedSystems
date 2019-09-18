#include <iostream>
#include <vector>
#include <string>

using namespace std;

class Individual{
	vector<double*> W;
	double fitness;
	string* activations;

	public:
		Individual(vector<double*> W, double fitness, string* activations){
			W = W;
			fitness = fitness;
			activations = activations;
		}
};