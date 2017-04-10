#include <vector>

using namespace std;

struct Connection {
   double weight;
};

class Neuron {
private:
   int index;
   double output;
   vector<Connection> outputWeights;
public:
   Neuron(int numOutputs, int _index);
   float getOutput();
   void setOuput();
   vector<Connection> getOutputWeights();
};

Neuron::Neuron(int numOutputs, int _index) {
   for (int connection = 0; connection < numOutputs; connection++) {
      Connection c = Connection();
      c.weight = 0.1; // Change this to be a random value
      outputWeights.push_back(c);
   }
   index = _index;
}

vector<Connection> Neuron::getOutputWeights() {
   vector<Connection> _outputWeights;
   for (int weight = 0; weight < outputWeights.size(); weight++) {
      _outputWeights.push_back(outputWeights[weight]);
   }
   
   return _outputWeights;
}