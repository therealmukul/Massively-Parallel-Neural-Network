#include <iostream>
#include <vector>
#include <mpi.h>
#include "Neuron.cpp"

using namespace std;

//  Global Variables
int worldSize;
int myRank;


int main(int argc, char *argv[]) {

   MPI_Init(&argc, &argv);
   MPI_Comm_size( MPI_COMM_WORLD, &worldSize);
   MPI_Comm_rank( MPI_COMM_WORLD, &myRank);

   Neuron *testNeuron = new Neuron(4, 1);
   vector<Connection> outputWeights = testNeuron->getOutputWeights();

   for (int i = 0; i < outputWeights.size(); i++) {
      cout << outputWeights[i].weight << endl;
   }
   
   MPI_Finalize();
   return 0;

}