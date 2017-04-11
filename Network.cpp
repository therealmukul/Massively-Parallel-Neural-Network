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
public:
   Network();
   void addLayer(string _type, int _size);
   void initializeNetwork();
   void printNetworkInfo();  // For debugging purposes
};

Network::Network() {
   // Not sure what will need to be added here. Nothing for now.
}

/*
   Add layers to define the topology of the network
*/
void Network::addLayer(string _type, int _size) {
   LayerTopology lyrTop = LayerTopology();
   lyrTop.size = _size;
   lyrTop.type = _type;
   networkTopology.push_back(lyrTop);
}

/*
   Once all layers have been added construct the topology of the network,
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

void Network::printNetworkInfo() {
   cout << "----------------------" << endl;
   for (int i = 0; i < layers.size(); i++) {
      string layerType = layers[i].getType();
      int layerSize = layers[i].getSize();
      printf("Type: %s Size: %d\n", layerType.c_str(), layerSize);
   }
   cout << "----------------------" << endl;
}



