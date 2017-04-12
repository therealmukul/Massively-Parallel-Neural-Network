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
   vector<Layer> layers;
   vector<LayerTopology> networkTopology;
   vector<vector<float> > inputData;
   vector<vector<float> > outputData;
public:
   Network();
   void addLayer(const string &_type, const int &_size);
   void initializeNetwork();
   void loadTestingInputData(const string &inputDataLoc);
   void loadTestingOutputData(const string &outputDataLoc, const int &numClasses);
   void printNetworkInfo();  // For debugging purposes
};

Network::Network() {
   // Not sure what will need to be added here. Nothing for now.
}

/*
   Add layers to define the topology of the network
*/
void Network::addLayer(const string &_type, const int &_size) {
   LayerTopology lyrTop = LayerTopology();
   lyrTop.size = _size;
   lyrTop.type = _type;
   networkTopology.push_back(lyrTop);
}

/*
   Once all layers have been added to construct the topology of the network,
   this function should be called to actaully build/initialize the network.
*/
void Network::initializeNetwork() {
   for (int currentLayer = 0; currentLayer < networkTopology.size(); currentLayer++) {
      int layerSize = networkTopology[currentLayer].size;
      int layerIndex = currentLayer + 1; // Starting numbering from "1"
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

   ifstream infile(inputDataLoc);
   string line;
   vector<float> sample;
   int inputSize = networkTopology[0].size;

   while (getline(infile, line)) {
      for (int c = 0; c < line.size(); c++) {
         char dataPoint = line[c];
         if (dataPoint != ',') {
            sample.push_back(((float)dataPoint - 48.0));
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
   ifstream infile(outputDataLoc);
   string c;
   vector<float> onehot(numClasses, 0.0);

   while (getline(infile, c)) {
      int dataPoint = stoi(c);
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

void Network::printNetworkInfo() { 
   cout << "----------------------" << endl;
   for (int i = 0; i < layers.size(); i++) {
      string layerType = layers[i].getType();
      int layerSize = layers[i].getSize();
      printf("Type: %s Size: %d\n", layerType.c_str(), layerSize);
   }
   cout << "----------------------" << endl;
}



