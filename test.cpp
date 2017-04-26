#include <iostream>
#include <mpi.h>

using namespace std;

int main(int argc, char *argv[]) {

   int myRank;
   int worldSize;

   MPI_Init(&argc, &argv);
   MPI_Comm_size( MPI_COMM_WORLD, &worldSize);
   MPI_Comm_rank( MPI_COMM_WORLD, &myRank);

   int *rbuf;
   int write = 0;
   if (myRank == 0) {
      cout << "Rank " << myRank << " allocated read buffer" << endl;
      rbuf = (int*)malloc(10*sizeof(int));
      cout << "Gather in rank " << myRank << endl;
      // MPI_Gather(&write, 1, MPI_INT, rbuf, 1, MPI_INT, 0, MPI_COMM_WORLD);

   } else {
      cout << "Gather in rank " << myRank << endl;
      write = 5;
      MPI_Gather(&write, 1, MPI_INT, NULL, 1, MPI_INT, 0, MPI_COMM_WORLD);
   }




   if (myRank == 0) {
      cout << rbuf[0] << endl;
   }



   MPI_Finalize();
   return 0;

}
