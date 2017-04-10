// Class to represent a Layer in the Neural Network
using namespace std;

class Layer {
private:
   int size;
   string type;
   vector<Neuron> neurons;
   Neuron *ghostNeuronTop;
   Neuron *ghostNeuronBottom;
public:
   Layer(int _size, int numNeuronsInNextLayer, string _type);
   string getType();
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
Layer::Layer(int _size, int numNeuronsInNextLayer, string _type) {
   size = _size;
   type = _type;
      
   for (int neuronIndex = 0; neuronIndex < size; neuronIndex++) {
      Neuron neuron = Neuron(numNeuronsInNextLayer, (neuronIndex + 1));
      neurons.push_back(neuron);
   }
   ghostNeuronTop = new Neuron(numNeuronsInNextLayer, 0);
   ghostNeuronTop = new Neuron(numNeuronsInNextLayer, 0);
}

string Layer::getType() {
   return type;
}

vector<Neuron> Layer::getNeurons() {
   return neurons;
}