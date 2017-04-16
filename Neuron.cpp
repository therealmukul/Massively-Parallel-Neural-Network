/*
   Class to represent a Neuron on the Neural Network
*/

using namespace std;

struct Connection {
   double weight;
};

class Neuron {
private:
   int index;
   double output;
   bool isGhost;
   vector<Connection> outputWeights;
public:
   Neuron(int numOutputs, int _index);
   void setOutput(double value);
   double getOutput();
   int getIndex();
   vector<Connection> getOutputWeights();
   void feedForward(vector<Neuron> &prevLayerNeurons);
};

/*
   Construct a single Neuron object

   Input: numOutputs
      Is the number of outgoing connections a neuron has.
      Will be the number of neurons in the next layer.
   Input: _index
      Integer value used to identify the neuron in the layer it belongs to.

   Return: Neuron object
*/
Neuron::Neuron(int numOutputs, int _index) {
   for (int connection = 0; connection < numOutputs; connection++) {
      Connection c = Connection();
      c.weight = 0.1; // Change this to be a random value
      outputWeights.push_back(c);
      output = 0.0;
   }
   if (index == -1) {
      isGhost = true;
   } else {
       isGhost = false;
   }
   index = _index;
}

void Neuron::setOutput(double value) {
   output = value;
}

double Neuron::getOutput() {
   return output;
}

vector<Connection> Neuron::getOutputWeights() {
   vector<Connection> _outputWeights;
   for (int weight = 0; weight < outputWeights.size(); weight++) {
      _outputWeights.push_back(outputWeights[weight]);
   }
   
   return _outputWeights;
}

int Neuron::getIndex() {
   return index;
}

void Neuron::feedForward(vector<Neuron> &prevLayerNeurons) {
   double sum = 0.0;
   for (int i = 0; i < prevLayerNeurons.size(); i++) {
      sum += prevLayerNeurons[i].getOutput() * 
             prevLayerNeurons[i].getOutputWeights()[index].weight;
   }
   output = sum;
   // cout << "   Neuron: " << index << " Output: " << sum << endl;

   
}









