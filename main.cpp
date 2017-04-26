#include <iostream>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <string>
#include <mpi.h>
#include <unistd.h>
#include <math.h>

//  Global Variables
int worldSize;
int myRank;

double *globalData = NULL;
double localData[2];
int outputsPerRank;

#include "Neuron.cpp"
#include "Layer.cpp"
#include "Network.cpp"

using namespace std;



int main(int argc, char *argv[]) {

   MPI_Init(&argc, &argv);
   MPI_Comm_size( MPI_COMM_WORLD, &worldSize);
   MPI_Comm_rank( MPI_COMM_WORLD, &myRank);

   int numInputs = 3;
   int numHidden = 4;
   int numOutputs = 4;

   int size = numOutputs + (numOutputs / (worldSize - 1));
   outputsPerRank = numOutputs / (worldSize - 1);

   // Construct a NN with 1 input layer, 1 hidden layer, and 1 output layer
   Network net = Network();
   net.addLayer("input", numInputs);  // 3 input neurons
   net.addLayer("hidden", numHidden/(worldSize - 1)); // 12 hidden neurons
   net.addLayer("output", numOutputs/(worldSize - 1)); // 4 output neurons
   net.initializeNetwork(size);


   net.loadTestingInputData("testInput.txt");
   net.loadTestingOutputData("testLabels.txt", 4);


   // Rank 0 is the master rank
   if (myRank == 0) {
      net.printNetworkInfo();

   }

   int iterations = 1;
   // Perform forward propogation for the specified number of iterations
   for (int i = 0; i < iterations; i++) {
      net.forwardPropagation();
      net.computeLoss(size);
      if (myRank == 0) {
         // printf("Iter: %d Loss: %f\n", i, loss);
      }
      net.backwardPropagation();
   }




   MPI_Finalize();
   return 0;

}
