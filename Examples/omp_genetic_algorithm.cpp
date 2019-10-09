#include <bits/stdc++.h> 
#include <iostream>
#include <math.h>
#include <vector>
#include <random>
#include <ctime>
#include <utility>
#include "omp.h"

#define N_THREADS 4

using namespace std;


double rastrigin(int A, int n, double x[]){
	double f = A * n;

	for(int i = 0; i < n; i++)
		f = f + (x[i]*x[i] - A * cos(2.0 * M_PI * x[i]));
	return f;
}


class Individual{
public:
	int dims;
	double fitness;
	double* chromosome;

	Individual(double*, double, int);
	Individual(const Individual&);
	
	Individual& operator=(const Individual& ind){
		dims = ind.dims;
		fitness = ind.fitness;
	
		chromosome = new double[dims];
		memcpy(chromosome, ind.chromosome, sizeof(double) * dims);
	}

	~Individual();
	void calculate_fitness(int);
};


Individual::Individual(double* chromosome, double fitness, int dims)
	:chromosome(chromosome),
	 fitness(fitness),
	 dims(dims){}


Individual::Individual(const Individual& ind){
	dims = ind.dims;
	fitness = ind.fitness;
	
	chromosome = new double[dims];
	memcpy(chromosome, ind.chromosome, sizeof(double) * dims);
}


Individual::~Individual(){
	delete []chromosome;
}


void Individual::calculate_fitness(int A){
	fitness = rastrigin(A, dims, chromosome);
}


double* initialize_individual(int n, double lim_max, double lim_min){
	double r;
	double* x = new double[n];

	static default_random_engine generator; 
	uniform_real_distribution<double> distribution(lim_min, lim_max); 
	
	for(int i = 0; i < n; ++i){
		r = distribution(generator);
		x[i] = r;
	}

	return x;
}


vector<Individual> initialize_population(int n, int P, double lim_max, double lim_min){
	vector<Individual> pop;

	for(int i = 0; i < P; ++i){
		double* x = initialize_individual(n, lim_max, lim_min);

		Individual ind(x, -1.0, n);
		pop.push_back(ind);
	}

	return pop;
}


bool compare_fitness(Individual i1, Individual i2){
	return i1.fitness < i2.fitness;
}


vector<Individual> rank_selection(vector<Individual> pop, int parents_size){
	vector<Individual> parents;
	double r, p;
	sort(pop.begin(), pop.end(), compare_fitness);

	static default_random_engine generator;
	uniform_real_distribution<double> distribution(0.0, 1.0);

	for(int i = 0; i < parents_size; ++i){
		r = distribution(generator);
		
		for(int j = 0; j < pop.size(); ++j){
			Individual ind = pop[j];
			p = (j + 1.0) / pop.size();

			if(r <= p){
				parents.push_back(ind);
				break;
			}
		}
	}

	return parents;
}


pair<Individual, Individual> crossover(Individual parent1, Individual parent2, int n){
	int crosspoint = rand() % (n - 1) + 1;
	double *c1 = new double[n], *c2 = new double[n];

	for(int i = 0; i < n; ++i){
		c1[i] = i < crosspoint ? parent1.chromosome[i] : parent2.chromosome[i];
		c2[i] = i < crosspoint ? parent2.chromosome[i] : parent1.chromosome[i];
	}

	Individual child1(c1, -1.0, n);
	Individual child2(c2, -1.0, n);

	pair<Individual, Individual> offspring(child1, child2);

	return offspring;
}


void mutation(Individual *child, int n){
	int pos = rand() % n;
	const double mean = 0.0, sigma = 0.1;
	
	static default_random_engine generator;
	normal_distribution<double> distribution(mean, sigma);

	child -> chromosome[pos] += distribution(generator);
}


pair<Individual, Individual> steady_state(vector<Individual> pool){
	sort(pool.begin(), pool.end(), compare_fitness);
	pair<Individual, Individual> new_gen(pool[0], pool[1]);
	return new_gen;
}


void print_best(vector<Individual> pop){
	double min = pop[0].fitness;
	for(Individual &ind: pop)
		if(ind.fitness < min)
			min = ind.fitness;

	cout << "Fitness of best individual: " << min << endl;
}


int main(){
	srand(time(0));

	int n = 10, parents_size = 2000;
	int A = 10.0, P = 10000, T = 150;
	double lim_min = -5.12, lim_max = 5.12;

	vector<Individual> parents, new_pop, pop = initialize_population(n, P, lim_max, lim_min);
	new_pop = pop;

	//#pragma omp parallel for num_threads(N_THREADS)
		for(int i = 0; i < P; ++i){
			pop[i].calculate_fitness(A);
		}

	for(int t = 1; t <= T; ++t){
		parents = rank_selection(pop, parents_size);

		#pragma omp parallel for num_threads(N_THREADS)
			for(int i = 0; i < P; i += 2){
				Individual parent1 = parents[rand() % parents_size];
				Individual parent2 = parents[rand() % parents_size];

				pair<Individual, Individual> offspring = crossover(parent1, parent2, n);			
				Individual child1 = offspring.first;
				Individual child2 = offspring.second;

				mutation(&child1, n); mutation(&child2, n);
				child1.calculate_fitness(A); child2.calculate_fitness(A);

				vector<Individual> pool {parent1, parent2, child1, child2};

				pair<Individual, Individual> new_gen = steady_state(pool);

				new_pop[i] = new_gen.first;
				new_pop[i + 1] = new_gen.second;

				//#pragma omp critical
				//{
					//new_pop.push_back(new_gen.first);
					//new_pop.push_back(new_gen.second);
				//}
			}

		pop = new_pop;

		if(t % 10 == 0){
			cout << endl << "Generation: " << t << endl;
			print_best(pop);
		}

		// /new_pop.clear();
	}

	pop.clear();
	return 0;
}