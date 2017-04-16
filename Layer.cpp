/*
   Class to represent a Layer in the Neural Network
*/

using namespace std;

class Layer {
private:
   int index;
   int size;
   string type;
   vector<Neuron> neurons;
   Neuron *ghostNeuronTop;
   Neuron *ghostNeuronBottom;
public:
   Layer(const int &_size, const int &numNeuronsInNextLayer, const string &_type, const int &_index);
   void setOutputValueForNeuronAtIndex(int index, double _outputValue);
   string getType();
   int getSize();
   vector<Neuron> getNeurons();
   void feedForward(Layer &prevLayer);
};

/*
   Contruct a single layer in the network.

   Input: _size
      The number of neurons in the layer.
   Input: numNeuronsInNextLayer
      The number of neurons in the next layer. 
      Is used to determine the number of outgoing connections 
      each neuron in the layer will have.
   Input: _type
      An identifier for the type of layer (input, hidden, output)

   Return: Layer object
*/
Layer::Layer(const int &_size, const int &numNeuronsInNextLayer, const string &_type, const int &_index) {
   size = _size;
   type = _type;
   index = _index;
      
   for (int neuronIndex = 0; neuronIndex < size; neuronIndex++) {
      if (type == "output") {
         // Output neurons have no outgoing connections
         Neuron neuron = Neuron(0, (neuronIndex));
         neurons.push_back(neuron);
      } else {
         Neuron neuron = Neuron(numNeuronsInNextLayer, (neuronIndex));
         neurons.push_back(neuron);
      }
      
   }

   if (type == "output") {
      // Ghost neurons in the output layer will have no outgoing connections
      ghostNeuronTop = new Neuron(0, -1);
      ghostNeuronBottom = new Neuron(0, -1);   
   } else {
      ghostNeuronTop = new Neuron(numNeuronsInNextLayer, -1);
      ghostNeuronBottom = new Neuron(numNeuronsInNextLayer, -1);
   } 
}

string Layer::getType() {
   return type;
}

int Layer::getSize() {
   return size;
}

vector<Neuron> Layer::getNeurons() {
   return neurons;
}

void Layer::setOutputValueForNeuronAtIndex(int index, double _outputValue) {
   neurons[index].setOutput(_outputValue);
}

void Layer::feedForward(Layer &prevLayer) {
   vector<Neuron> prevLayerNeurons = prevLayer.getNeurons();

   for (int neuron = 0; neuron < neurons.size(); neuron++) {
      neurons[neuron].feedForward(prevLayerNeurons);
   }

   double ghostTopOutput;
   double ghostBottomOutput;
   double firstNeuronOutput = neurons[0].getOutput();
   double lastNeuronOutput = neurons[neurons.size() - 1].getOutput();

   MPI_Status status;
   MPI_Request rcvTopRequest;
   MPI_Request rcvBottomRequest;
   MPI_Request sndBottomRequest;
   MPI_Request sndTopRequest;

   if (type != "output" and worldSize > 1) {
      if (myRank == 0) { // FIRST RANK

         // R_0 (firstNeuronOutput) -> R_N (ghostBottom)
         MPI_Isend(&firstNeuronOutput, 1, MPI_DOUBLE, (worldSize - 1), 0, MPI_COMM_WORLD, &sndBottomRequest);
         MPI_Wait(&sndBottomRequest, &status);
         // printf("Rank %d sent value %f to rank %d\n", myRank, firstNeuronOutput, (worldSize - 1));

         // R_0 (lastNeuronOuput) -> R_1 (ghostTop)
         MPI_Isend(&lastNeuronOutput, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, &sndTopRequest);
         MPI_Wait(&sndTopRequest, &status);
         // printf("Rank %d sent value %f to rank %d\n", myRank, lastNeuronOutput, 1);

         // ghostBottomOutput <- R_1 (firstNeuronOutput)
         int ret = MPI_Irecv(&ghostBottomOutput, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, &rcvBottomRequest);
         MPI_Wait(&rcvBottomRequest, &status);
         if (ret == MPI_SUCCESS) {
            // printf("Rank %d received value %f from rank 1\n", myRank, ghostBottomOutput);
         }

         // ghostTopOutput <- R_N (lastNeuronOutput)
         int ret2 = MPI_Irecv(&ghostTopOutput, 1, MPI_DOUBLE, (worldSize - 1), 0, MPI_COMM_WORLD, &rcvTopRequest);
         MPI_Wait(&rcvTopRequest, &status);
         if (ret2 == MPI_SUCCESS) {
            // printf("Rank %d received value %f from rank 1\n", myRank, ghostTopOutput);
         }


      } else if (myRank == worldSize - 1) { // LAST RANK

         // R_Last (firstNeuronOutput) -> R_Last-1 (ghostBottom)
         MPI_Isend(&firstNeuronOutput, 1, MPI_DOUBLE, (myRank - 1), 0, MPI_COMM_WORLD, &sndBottomRequest);
         MPI_Wait(&sndBottomRequest, &status);
         // printf("Rank %d sent value %f to rank %d\n", myRank, firstNeuronOutput, (myRank - 1));

         // R_Last (lastNeuronOutput) -> R_0 (ghostTop)
         MPI_Isend(&lastNeuronOutput, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &sndTopRequest);
         MPI_Wait(&sndTopRequest, &status);
         // printf("Rank %d sent value %f to rank %d\n", myRank, lastNeuronOutput, 0);

         // ghostBottomOutput <- R_0 (firstNeuronOutput)
         int ret = MPI_Irecv(&ghostBottomOutput, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &rcvBottomRequest);
         MPI_Wait(&rcvBottomRequest, &status);
         if (ret == MPI_SUCCESS) {
            // printf("Rank %d received value %f from rank 0\n", myRank, ghostBottomOutput);
         }

         // ghostTopOutput <- R_0 (firstNeuronOutput)
         int ret2 = MPI_Irecv(&ghostTopOutput, 1, MPI_DOUBLE, (myRank - 1), 0, MPI_COMM_WORLD, &rcvTopRequest);
         MPI_Wait(&rcvTopRequest, &status);
         if (ret2 == MPI_SUCCESS) {
            // printf("Rank %d received value %f from rank %d\n", myRank, ghostBottomOutput, (myRank - 1));
         }
         
      } else { // ALL RANKS INBETWEEN

         // R_r (firstNeuronOutput) -> R_r-1 (ghostBottom)
         MPI_Isend(&firstNeuronOutput, 1, MPI_DOUBLE, (myRank - 1), 0, MPI_COMM_WORLD, &sndBottomRequest);
         MPI_Wait(&sndBottomRequest, &status);
         // printf("Rank %d sent value %f to rank %d\n", myRank, firstNeuronOutput, (myRank - 1));

         // R_r (lastNeuronOutput) -> R_r+1 (ghostTop)
         MPI_Isend(&lastNeuronOutput, 1, MPI_DOUBLE, (myRank + 1), 0, MPI_COMM_WORLD, &sndTopRequest);
         MPI_Wait(&sndTopRequest, &status);
         // printf("Rank %d sent value %f to rank %d\n", myRank, lastNeuronOutput, (myRank + 1));

         // ghostBottomOutput <- R_r+1 (lastNeuronOutput)
         int ret = MPI_Irecv(&ghostBottomOutput, 1, MPI_DOUBLE, (myRank + 1), 0, MPI_COMM_WORLD, &rcvBottomRequest);
         MPI_Wait(&rcvBottomRequest, &status);
         if (ret == MPI_SUCCESS) {
            // printf("Rank %d received value %f from rank %d\n", myRank, ghostBottomOutput, (myRank + 1));
         }

         // ghostTopOutput <- R_r-1 (lastNeuronOutput)
         int ret2 = MPI_Irecv(&ghostTopOutput, 1, MPI_DOUBLE, (myRank - 1), 0, MPI_COMM_WORLD, &rcvTopRequest);
         MPI_Wait(&rcvTopRequest, &status);
         if (ret2 == MPI_SUCCESS) {
            // printf("Rank %d received value %f from rank %d\n", myRank, ghostTopOutput, (myRank - 1));
         }
      }      
   }

   // Set the Ghost Neuron Outputs for each layer
   cout << "Rank " << myRank << " " << type << endl;
   cout << "GhostTopOutput " << ghostNeuronTop->getOutput() << endl;
   cout << "GhostBottomOutput " << ghostNeuronBottom->getOutput() << endl;
     
}