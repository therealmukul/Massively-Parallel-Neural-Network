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
   string getType();
   int getSize();
   vector<Neuron> getNeurons();
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
         Neuron neuron = Neuron(0, (neuronIndex + 1));
         neurons.push_back(neuron);
      } else {
         Neuron neuron = Neuron(numNeuronsInNextLayer, (neuronIndex + 1));
         neurons.push_back(neuron);
      }
      
   }

   if (type == "output") {
      // Ghost neurons in the output layer will have no outgoing connections
      ghostNeuronTop = new Neuron(0, 0);
      ghostNeuronTop = new Neuron(0, 0);   
   }
   ghostNeuronTop = new Neuron(numNeuronsInNextLayer, 0);
   ghostNeuronTop = new Neuron(numNeuronsInNextLayer, 0);
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