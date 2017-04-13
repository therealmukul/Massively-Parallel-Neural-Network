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
      ghostNeuronTop = new Neuron(0, -1);   
   }
   ghostNeuronTop = new Neuron(numNeuronsInNextLayer, -1);
   ghostNeuronTop = new Neuron(numNeuronsInNextLayer, -1);
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
   // for (int i = 0; i < prevLayerNeurons.size(); i++) {
   //    cout << prevLayerNeurons[i].getOutput() << endl;
   // }
   for (int neuron = 0; neuron < neurons.size(); neuron++) {
      neurons[neuron].feedForward(prevLayerNeurons);
   }
}