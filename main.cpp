#include <iostream>
#include <vector>
#include <string>
#include <mpi.h>

#include "Neuron.cpp"
#include "Layer.cpp"

using namespace std;

//  Global Variables
int worldSize;
int myRank;


int main(int argc, char *argv[]) {

   MPI_Init(&argc, &argv);
   MPI_Comm_size( MPI_COMM_WORLD, &worldSize);
   MPI_Comm_rank( MPI_COMM_WORLD, &myRank);

   Layer l1 = Layer(3, 5, "input");
   cout << l1.getType() << endl;
   vector<Neuron> l1Neurons = l1.getNeurons();
   for (auto neuron : l1Neurons) {
      cout << neuron.getIndex() << endl;
   }

   MPI_Finalize();
   return 0;

}