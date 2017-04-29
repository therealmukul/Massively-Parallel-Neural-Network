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

double *globalData = NULL;
// double localData[5]; // numOutputs / worldSize - 1
double *localData = NULL;
int outputsPerRank;

#include "Neuron.cpp"
#include "Layer.cpp"
#include "Network.cpp"

using namespace std;

int gcd(int a, int b) {
   return b == 0 ? a : gcd(b, a % b);
}

int main(int argc, char *argv[]) {

   MPI_Init(&argc, &argv);
   MPI_Comm_size( MPI_COMM_WORLD, &worldSize);
   MPI_Comm_rank( MPI_COMM_WORLD, &myRank);

   int numInputs = 100;
   int numHidden = 400;
   int numOutputs = 100;

   // Check that the number of ranks does not exceed the maximum allowed
   if(myRank == 0){
      if(worldSize > (gcd(numHidden,numOutputs)+1)){
         printf("Error: Too many ranks; number of ranks can be at max GCD(numHidden,numOutputs)+1\n");
         MPI_Abort(MPI_COMM_WORLD,1);
      }
   }

   int size = numOutputs + (numOutputs / (worldSize - 1));
   outputsPerRank = numOutputs / (worldSize - 1);

   // Construct a NN with 1 input layer, 1 hidden layer, and 1 output layer
   Network net = Network();
   net.addLayer("input", numInputs);  // 3 input neurons
   net.addLayer("hidden", numHidden/(worldSize - 1)); // 12 hidden neurons
   net.addLayer("output", numOutputs/(worldSize - 1)); // 4 output neurons
   net.initializeNetwork(size);


   net.loadTestingInputData("genTestInput.txt");
   net.loadTestingOutputData("genTestLabels.txt", numOutputs);


   // Rank 0 is the master rank
   if (myRank == 0) {
      net.printNetworkInfo();

   }

   int iterations = 20;
   // Perform forward propogation for the specified number of iterations
   for (int i = 0; i < iterations; i++) {
      net.forwardPropagation();
      if (myRank == 1) {
         // cout << i << endl;
      }

      // net.testUpdate();
      double loss = net.computeLoss(size);
      if (myRank == 0) {
         printf("Iter: %d Loss: %f\n", i, loss);
      }
      // if (myRank == 1) {
      //    cout << "Rank 1 Input Layer BEFORE" << endl;
      //    net.printLayerWeights(0);
      // }
      net.backwardPropagation();
      // if (myRank == 1) {
      //    cout << "Rank 1 Input Layer AFTER" << endl;
      //    net.printLayerWeights(0);
      // }
      // usleep(1000000);


   }

   MPI_Finalize();
   return 0;

}
