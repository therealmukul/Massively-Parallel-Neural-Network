/*
   Class to represent the entire Artifical Neural Network
*/

using namespace std;

struct LayerTopology {
   int size;
   string type;
};

class Network {
private:
   int sampleIndex;
   vector<Layer> layers;
   vector<LayerTopology> networkTopology;
   vector<vector<double> > inputData;
   vector<vector<double> > outputData;
   vector<double> targetOutput;
   vector<double> outputGradients;
   vector<double> multiclassSigmoid(vector<double> &yHat);
public:
   Network();
   void addLayer(const string &_type, const int &_size);
   void initializeNetwork(int size);
   void loadTestingInputData(const string &inputDataLoc);
   void loadTestingOutputData(const string &outputDataLoc, const int &numClasses);
   void forwardPropagation();
   void backwardPropagation();
   double computeLoss(int size);

   void printNetworkInfo();  // For debugging purposes
};

// Private Methods
vector<double> Network::multiclassSigmoid(vector<double> &yHat) {
   double totalExp = 0;
   for (int i = 0; i < yHat.size(); i++) {
      totalExp += exp(yHat[i]);
   }
   for (int j = 0; j < yHat.size(); j++) {
      yHat[j] = exp(yHat[j]) / totalExp;
   }
   return yHat;
}

Network::Network() {
   sampleIndex = 0;
}

/*
   Add layers to define the topology of the network
*/
void Network::addLayer(const string &_type, const int &_size) {
   if(_type == "input" || _type == "hidden" || _type == "output"){
      LayerTopology lyrTop = LayerTopology();
      lyrTop.size = _size;
      lyrTop.type = _type;
      networkTopology.push_back(lyrTop);
   }else{
      if(myRank == 0){
         cout << "Error: "<< _type <<" is not a valid layer type. Valid layer types are input, hidden, and output." << "\n";
         MPI_Abort(MPI_COMM_WORLD,1);
      }
   }
}

/*
   Once all layers have been added to construct the topology of the network,
   this function should be called to actaully build/initialize the network.
*/
void Network::initializeNetwork(int size) {
   for (int currentLayer = 0; currentLayer < networkTopology.size(); currentLayer++) {
      int layerSize = networkTopology[currentLayer].size;
      int layerIndex = currentLayer;
      string layerType = networkTopology[currentLayer].type;

      if (currentLayer <= networkTopology.size() - 1) {
         int numNeuronsInNextLayer = networkTopology[currentLayer + 1].size;
         Layer newLayer = Layer(layerSize, numNeuronsInNextLayer, layerType, layerIndex);
         layers.push_back(newLayer);
      } else {
         Layer newLayer = Layer(layerSize, 0, layerType, layerIndex);
         layers.push_back(newLayer);
      }
   }

   globalData = (double*)malloc(size * sizeof(double));
}

/*
   Reads the input file line by line and stores each input sample into inputData.
   Each input sample is split into a vector. Ex Input 1,0,1 --> Vec<1, 0, 1>

   Input: inputDataLoc
      Location of the testing input data file.

   TODO: This function can be made more efficent. I basically brute forced it for now.
         Some sort of error checking should be added to make sure each input sample
         contains the equivilant number of inputs as specifed in the input layer.

         Just realized that its not smart to load all input data into memory *face-palm*
         OK for debugging small examples. Should change this later so that its part of
         forward propgation.
*/
void Network::loadTestingInputData(const string &inputDataLoc) {

   ifstream infile(inputDataLoc.c_str());
   string line;
   vector<double> sample;
   int inputSize = networkTopology[0].size;

   while (getline(infile, line)) {
      for (int c = 0; c < line.size(); c++) {
         char dataPoint = line[c];
         if (dataPoint != ',') {
            sample.push_back(((double)dataPoint - 48.0));
         }
      }
      inputData.push_back(sample);
      sample.clear();
   }

   // // Print out the input data -- DEBUG
   // cout << "Input Data" << endl;
   // for (int i = 0; i < inputData.size(); i++) {
   //    for (int j = 0; j < inputData[i].size(); j++) {
   //       cout << inputData[i][j] << " ";
   //    }
   //    cout << endl;
   // }
}

/*
   Read the expected ouput file. Convert eac label into onehot encoding and store it in
   outputData. Onehot enoding example given 4 classes: label 3 --> Vec<0, 0, 1, 0>

   Input: outputDataLoc
      Location of testing labels files
   Input: numClasses
      The number of total classes which the net is classifying between.

   TODO: Just realized that its not smart to load all label data into memory *face-palm*
         OK for debugging small examples. Should change this later so that its part of
         forward propgation.
*/
void Network::loadTestingOutputData(const string &outputDataLoc, const int &numClasses) {
   ifstream infile(outputDataLoc.c_str());
   string c;
   vector<double> onehot(numClasses, 0.0);

   while (getline(infile, c)) {
      int dataPoint = atoi(c.c_str());
      onehot[dataPoint - 1] = 1.0;
      outputData.push_back(onehot);
      onehot[dataPoint - 1] = 0.0;
   }

   // // Print out the Onehot encoding of the output labels -- DEBUG
   // cout << "Onehot Encoding of output labels" << endl;
   // for (int i = 0; i < outputData.size(); i++) {
   //    for (int j = 0; j < outputData[i].size(); j++) {
   //       cout << outputData[i][j] << " ";
   //    }
   //    cout << endl;
   // }
}

void Network::forwardPropagation() {
   // Select a sample from the input data
   int index = sampleIndex % inputData.size();
   vector<double> inputSample = inputData[index];

   if (myRank > 0) {

      // Feed the sample input values into the neurons in the input layer
      int inputLayerIndex = 0;
      for (int value = 0; value < inputSample.size(); value++) {
         layers[inputLayerIndex].setOutputValueForNeuronAtIndex(value, inputSample[value]);
      }

      // Forward Propogate
      for (int layerNum = 1; layerNum < layers.size(); layerNum++) {
         // cout << layers[layerNum].getType() << " Rank " << myRank << endl;
         Layer prevLayer = layers[layerNum - 1];
         layers[layerNum].feedForward(prevLayer);
         // MPI_Barrier(MPI_COMM_WORLD);
      }

   }
   sampleIndex++;
}

void Network::backwardPropagation() {

   if (myRank > 0) {
      //    printf("Rank %d has data: ", myRank);
      //    for (int j = 0; j < outputGradients.size(); j++) {
      //       cout << outputGradients[j] << " ";
      //    }
      //    cout << endl;


      // Assign output gradients to each neuron
      int numOutputs = networkTopology.back().size;
      int offset = (myRank * numOutputs) - numOutputs;
      int neuronIndex = 0;
      for (int i = offset; i < (offset + numOutputs); i++) {
         double gradient = outputGradients[i];
         layers.back().setNeuronGradientForNeuronAtIndex(neuronIndex, gradient);
         neuronIndex++;
      }

      // Calculate and assign gradients on hidden layers
      for (int layerNum = layers.size() - 2; layerNum > 0; layerNum--) {
         Layer &hiddenLayer = layers[layerNum];
         Layer &nextLayer = layers[layerNum + 1];
         hiddenLayer.calcHiddenGradients(nextLayer);
      }

      // Updated the weights
      for (int layerNum = layers.size() - 1; layerNum > 0; layerNum--) {
         Layer &currentLayer = layers[layerNum];
         Layer &prevLayer = layers[layerNum - 1];
         currentLayer.updateWeights(prevLayer);
      }

   }

}

double Network::computeLoss(int size) {
   int outputsPerRank = networkTopology.back().size;
   int ret = MPI_Allgather(&localData, outputsPerRank, MPI_DOUBLE, globalData, outputsPerRank, MPI_DOUBLE, MPI_COMM_WORLD);
   double loss = 0;
   if (ret == MPI_SUCCESS) {
      vector<double> yHat;
      for (int i = outputsPerRank; i < (outputsPerRank * (worldSize)); i++) {
         yHat.push_back(globalData[i]);
      }

      printf("Rank %d has data: ", myRank);
      for (int i = 0; i < yHat.size(); i++) {
         cout << yHat[i] << " ";
      }
      cout << endl;

      yHat = multiclassSigmoid(yHat);
      vector<double> yPred = outputData[sampleIndex - 1];
      vector<double> yGradient;

      for (int i = 0; i < yHat.size(); i++) {
         double gradVal = -1 * (yPred[i] - yHat[i]);
         yGradient.push_back(gradVal);
      }

      outputGradients = yGradient;
      targetOutput = yPred;

      for (int i = 0; i < yGradient.size(); i++) {
         loss += yGradient[i] * yGradient[i];
      }
      loss = loss / 2;
      // backwardPropagation();
   }

   return loss;
}


void Network::printNetworkInfo() {
   cout << "----------------------" << endl;
   cout << "Num Rank: " << worldSize << endl;
   cout << "Each rank other than master handles:" << endl;
   for (int i = 0; i < layers.size(); i++) {
      string layerType = layers[i].getType();
      int layerSize = layers[i].getSize();
      printf("  Type: %s Size: %d\n", layerType.c_str(), layerSize);
   }
   cout << "----------------------" << endl;
}
