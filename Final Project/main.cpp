#include <iostream>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <string>
#include <mpi.h>
#include <unistd.h>
#include <math.h>
#include <stdlib.h>

//  Global Variables
int worldSize;
int myRank;
double rankTime = 0.0;

double *globalData = NULL;
double *localData = NULL;
int outputsPerRank;

#include "Neuron.cpp"
#include "Layer.cpp"
#include "Network.cpp"

using namespace std;

// Utility function to make sure the maximum number of ranks is not violated.
int gcd(int a, int b) {
   return b == 0 ? a : gcd(b, a % b);
}

int main(int argc, char *argv[]) {

   MPI_Init(&argc, &argv);
   MPI_Comm_size( MPI_COMM_WORLD, &worldSize);
   MPI_Comm_rank( MPI_COMM_WORLD, &myRank);

   int numInputs = 2048;
   int numHidden1 = 32768;
   int numHidden2 = 8192;
   int numHidden3 = 4096;
   int numOutputs = 2048;

   double startTimeT = 0;
   double endTimeT = 0;
   double totalTimeT = 0;

   // Check that the number of ranks does not exceed the maximum allowed
   if(myRank == 0){
      if(worldSize > (gcd(numHidden1, numOutputs) + 1) || worldSize > (gcd(numHidden2, numOutputs) + 1) || worldSize > (gcd(numHidden3, numOutputs) + 1) ){
         printf("Error: Too many ranks; number of ranks can be at max GCD(numHidden,numOutputs)+1\n");
         MPI_Abort(MPI_COMM_WORLD,1);
      }
   }

   // Compute the number of output values per rank
   int size = numOutputs + (numOutputs / (worldSize - 1));
   outputsPerRank = numOutputs / (worldSize - 1);
   localData = (double*)malloc(outputsPerRank * sizeof(double));

   // Construct a NN with 1 input layer, 3 hidden layers, and 1 output layer
   Network net = Network();
   net.addLayer("input", numInputs);
   net.addLayer("hidden", numHidden1/(worldSize - 1));
   net.addLayer("hidden", numHidden2/(worldSize - 1));
   net.addLayer("hidden", numHidden3/(worldSize - 1));
   net.addLayer("output", numOutputs/(worldSize - 1));
   net.initializeNetwork(size);

   // Load the testing data
   net.loadTestingInputData("genTestInput.txt");
   net.loadTestingOutputData("genTestLabels.txt", numOutputs);

   // Print the network info
   if (myRank == 0) {
      net.printNetworkInfo();
   }

   // Used to determine performance
   if (myRank == 1) {
      startTimeT = MPI_Wtime();
   }

   int iterations = 20;
   // Train the network for the specified number of iterations
   for (int i = 0; i < iterations; i++) {
      double startTime = 0;
      double endTime = 0;
      double totalTime = 0;

      if (myRank == 1) {
         startTime = MPI_Wtime();
      }

      net.forwardPropagation();
      net.computeLoss(size);
      net.backwardPropagation();

      if (myRank == 1) {
         endTime = MPI_Wtime();
         totalTime = endTime - startTime;
         printf("Iter: %d Time: %f\n", i, totalTime);
      }
   }

   // Compute the total time for the number of iterations
   if (myRank == 1) {
      endTimeT = MPI_Wtime();
      totalTimeT = endTimeT - startTimeT;
      printf("Total Time: %f\n", totalTimeT);
   }

   MPI_Finalize();
   return 0;

}
