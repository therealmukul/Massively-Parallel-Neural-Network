A Massively Parallel Artificial Deep Neural Network Framework for IBM Blue Gene/Q
Mukul Surajiwale, Rabiul Chowdhury, Kyle Fawcett, Alex Giris
CSCI 4320: Parallel Programming Spring 2017
5/2/2017

Preloaded Model Info:
   Input:         2,048
   Hidden 1:      32,756
   Hidden 2:      8,192
   Hidden 3:      4,096
   Output:        2,048

   Total Weights: 377,487,360

How to run:
   Compile: mpic++ main.cpp -o run
   Run:     mpirun -n <num ranks> ./run

Expected Output:
   Model will train for 20 iterations.
   The time needed to complete each iteration will be displayed.

Notes:
   -  Our framework dedicates rank 0 to be the master rank to manage the overall
      network. Thus, if you want to run the model in sequential you must use run it
      with 2 ranks. If you want to run the model with 32 working ranks then you
      must run the model with 33 ranks to account for the master rank.

   -  The data provide is dummy generated data. It is used to test the scaling
      performance of the network.

Modifying Network:
   Adding more layers:
      net.addLayer("<type>", <num neurons in layer>)
      - You can add as many layers and neurons as you like.
      - There can only be one "input" and "output" layer.
