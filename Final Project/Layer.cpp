/*
   Class to represent a Layer in the Neural Network
*/

using namespace std;

class Layer {
private:
   int index;
   int size;
   string type;
   vector<Neuron*> neurons;
   void performGhostNeuronMsgPassing();
public:
   Neuron *ghostNeuronTop;
   Neuron *ghostNeuronBottom;

   Layer(const int &_size, const int &numNeuronsInNextLayer, const string &_type, const int &_index);
   void setOutputValueForNeuronAtIndex(int index, double _outputValue);
   string getType();
   int getSize();
   int getIndex();
   vector<Neuron*> getNeurons();
   void feedForward(Layer *prevLayer);
   void calcHiddenGradients(Layer *nextLayer);
   void updateWeights(Layer *prevLayer, int layerNum);
   void setNeuronGradientForNeuronAtIndex(int index, double gradient);
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
         Neuron *neuron = new Neuron(0, (neuronIndex));
         neurons.push_back(neuron);
      } else {
         Neuron *neuron = new Neuron(numNeuronsInNextLayer, (neuronIndex));
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

int Layer::getIndex() {
   return index;
}

vector<Neuron*> Layer::getNeurons() {
   return neurons;
}


void Layer::setOutputValueForNeuronAtIndex(int index, double _outputValue) {
   neurons[index]->setOutput(_outputValue);
}

// Parallel Forward Propgation function. Performs all the message passing required for the
// current layer.
void Layer::performGhostNeuronMsgPassing() {

   double startTimeRcv1 = 0;
   double endTimeRcv1 = 0;
   double totalTimeRcv1 = 0;

   double startTimeRcv2 = 0;
   double endTimeRcv2 = 0;
   double totalTimeRcv2 = 0;

   double ghostTopOutput;
   double ghostBottomOutput;
   double firstNeuronOutput = neurons[0]->getOutput();
   double lastNeuronOutput = neurons[neurons.size() - 1]->getOutput();

   MPI_Status status;
   MPI_Request rcvTopRequest;
   MPI_Request rcvBottomRequest;
   MPI_Request sndBottomRequest;
   MPI_Request sndTopRequest;


   if (type != "output" and worldSize > 2) {
      if (myRank == 1) { // FIRST RANK

         // R_0 (firstNeuronOutput) -> R_N (ghostBottom)
         MPI_Isend(&firstNeuronOutput, 1, MPI_DOUBLE, (worldSize - 1), 0, MPI_COMM_WORLD, &sndBottomRequest);
         MPI_Wait(&sndBottomRequest, &status);

         // R_0 (lastNeuronOuput) -> R_2 (ghostTop)
         MPI_Isend(&lastNeuronOutput, 1, MPI_DOUBLE, 2, 0, MPI_COMM_WORLD, &sndTopRequest);
         MPI_Wait(&sndTopRequest, &status);

         // ghostBottomOutput <- R_2 (firstNeuronOutput)
         int ret = MPI_Irecv(&ghostBottomOutput, 1, MPI_DOUBLE, 2, 0, MPI_COMM_WORLD, &rcvBottomRequest);
         MPI_Wait(&rcvBottomRequest, &status);
         if (ret == MPI_SUCCESS) {
            ghostNeuronBottom->setOutput(ghostBottomOutput);
         }

         // ghostTopOutput <- R_N (lastNeuronOutput)
         int ret2 = MPI_Irecv(&ghostTopOutput, 1, MPI_DOUBLE, (worldSize - 1), 0, MPI_COMM_WORLD, &rcvTopRequest);
         MPI_Wait(&rcvTopRequest, &status);
         if (ret2 == MPI_SUCCESS) {
            ghostNeuronTop->setOutput(ghostTopOutput);
         }

      } else if (myRank == worldSize - 1) { // LAST RANK

         // R_Last (firstNeuronOutput) -> R_Last-1 (ghostBottom)
         MPI_Isend(&firstNeuronOutput, 1, MPI_DOUBLE, (myRank - 1), 0, MPI_COMM_WORLD, &sndBottomRequest);
         MPI_Wait(&sndBottomRequest, &status);

         // R_Last (lastNeuronOutput) -> R_1 (ghostTop)
         MPI_Isend(&lastNeuronOutput, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, &sndTopRequest);
         MPI_Wait(&sndTopRequest, &status);

         // ghostBottomOutput <- R_1 (firstNeuronOutput)
         int ret = MPI_Irecv(&ghostBottomOutput, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, &rcvBottomRequest);
         MPI_Wait(&rcvBottomRequest, &status);
         if (ret == MPI_SUCCESS) {
            ghostNeuronBottom->setOutput(ghostBottomOutput);

         }

         // ghostTopOutput <- R_0 (firstNeuronOutput)
         int ret2 = MPI_Irecv(&ghostTopOutput, 1, MPI_DOUBLE, (myRank - 1), 0, MPI_COMM_WORLD, &rcvTopRequest);
         MPI_Wait(&rcvTopRequest, &status);
         if (ret2 == MPI_SUCCESS) {
            ghostNeuronTop->setOutput(ghostTopOutput);
         }

      } else { // ALL RANKS INBETWEEN

         // R_r (firstNeuronOutput) -> R_r-1 (ghostBottom)
         MPI_Isend(&firstNeuronOutput, 1, MPI_DOUBLE, (myRank - 1), 0, MPI_COMM_WORLD, &sndBottomRequest);
         MPI_Wait(&sndBottomRequest, &status);

         // R_r (lastNeuronOutput) -> R_r+1 (ghostTop)
         MPI_Isend(&lastNeuronOutput, 1, MPI_DOUBLE, (myRank + 1), 0, MPI_COMM_WORLD, &sndTopRequest);
         MPI_Wait(&sndTopRequest, &status);

         // ghostBottomOutput <- R_r+1 (lastNeuronOutput)
         startTimeRcv1 = MPI_Wtime();
         int ret = MPI_Irecv(&ghostBottomOutput, 1, MPI_DOUBLE, (myRank + 1), 0, MPI_COMM_WORLD, &rcvBottomRequest);
         MPI_Wait(&rcvBottomRequest, &status);
         if (ret == MPI_SUCCESS) {
            ghostNeuronBottom->setOutput(ghostBottomOutput);
            endTimeRcv1 = MPI_Wtime();
            totalTimeRcv1 = endTimeRcv1 - startTimeRcv1;
            rankTime += totalTimeRcv1;
         }

         // ghostTopOutput <- R_r-1 (lastNeuronOutput)
         startTimeRcv2 = MPI_Wtime();
         int ret2 = MPI_Irecv(&ghostTopOutput, 1, MPI_DOUBLE, (myRank - 1), 0, MPI_COMM_WORLD, &rcvTopRequest);
         MPI_Wait(&rcvTopRequest, &status);
         if (ret2 == MPI_SUCCESS) {
            ghostNeuronTop->setOutput(ghostTopOutput);
            endTimeRcv2 = MPI_Wtime();
            totalTimeRcv2 = endTimeRcv2 - startTimeRcv2;
            rankTime += totalTimeRcv2;

         }
      }
   }
}

void Layer::feedForward(Layer *prevLayer) {
   vector<Neuron*> prevLayerNeurons = prevLayer->getNeurons();
   for (int neuron = 0; neuron < neurons.size(); neuron++) {
      neurons[neuron]->feedForward(prevLayerNeurons, index, prevLayer->ghostNeuronTop, prevLayer->ghostNeuronBottom);
   }

   // Peform all the message passing for the current layer
   performGhostNeuronMsgPassing();

}

// Calculate the gradients of the hidden layer
void Layer::calcHiddenGradients(Layer *nextLayer) {
   vector<Neuron*> nextLayerNeurons = nextLayer->getNeurons();
   for (int i = 0; i < neurons.size(); i++) {
      neurons[i]->calcHiddenGradients(nextLayerNeurons, ghostNeuronTop, ghostNeuronBottom);
   }
}

// Utility function. Pretty obvious from the name what it does
void Layer::setNeuronGradientForNeuronAtIndex(int index, double gradient) {
   neurons[index]->setGradient(gradient);
}

// Part of gradient descent. Updates all the weights in the layer.
void Layer::updateWeights(Layer *prevLayer, int layerNum) {
   vector<Neuron*> prevLayerNeurons = prevLayer->getNeurons();
   for (int i = 0; i < neurons.size(); i++) {
      neurons[i]->updateWeights(prevLayerNeurons, ghostNeuronTop, ghostNeuronBottom, layerNum);
   }
}
