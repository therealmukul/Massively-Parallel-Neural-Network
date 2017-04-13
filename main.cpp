#include <iostream>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <string>
#include <mpi.h>

#include "Neuron.cpp"
#include "Layer.cpp"
#include "Network.cpp"

using namespace std;

//  Global Variables
int worldSize;
int myRank;

int main(int argc, char *argv[]) {

   MPI_Init(&argc, &argv);
   MPI_Comm_size( MPI_COMM_WORLD, &worldSize);
   MPI_Comm_rank( MPI_COMM_WORLD, &myRank);

   // Construct a NN with 1 input layer, 1 hidden layer, and 1 output layer
   Network net = Network();
   net.addLayer("input", 3);  // 3 input neurons
   net.addLayer("hidden", 12/worldSize); // 12 hidden neurons
   net.addLayer("output", 4/worldSize); // 4 output neurons
   net.initializeNetwork();
   

   net.loadTestingInputData("testInput.txt");
   net.loadTestingOutputData("testLabels.txt", 4);

   net.printNetworkInfo();


   int iterations = 2;
   // Perform forward propogation for the specified number of iterations
   for (int i = 0; i < iterations; i++) {
      net.forwardPropogation();
   }

   MPI_Finalize();
   return 0;

}