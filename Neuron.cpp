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
   void setOuput(double value);
   double getOutput();
   vector<Connection> getOutputWeights();
   int getIndex();
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
   }
   if (index == 0) {
      isGhost = true;
   } else {
       isGhost = false;
   }
   index = _index;
}

void Neuron::setOuput(double value) {
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