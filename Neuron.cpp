/*
   Class to represent a Neuron on the Neural Network
*/

using namespace std;

struct Connection {
   double weight;
   double deltaWeight;
};

class Neuron {
private:
   int index;
   double output;
   double gradient;
   double eta;
   bool isGhost;
   vector<Connection> outputWeights;
   double sigmoid(double x);
   double sigmoidDerivative(double x);
public:
   Neuron(int numOutputs, int _index);
   void setOutput(double value);
   void setGradient(double value);
   double getOutput();
   int getIndex();
   vector<Connection> getOutputWeights();
   void feedForward(vector<Neuron> &prevLayerNeurons, int layerIndex, Neuron *ghostNeuronTop, Neuron *ghostNeuronBottom);
   void calcHiddenGradients(vector<Neuron> &nextLayerNeurons, Neuron *ghostNeuronTop, Neuron *ghostNeuronBottom);
   void updateWeights(vector<Neuron> &prevLayerNeurons, Neuron *ghostNeuronTop, Neuron *ghostNeuronBottom);
   double sumDOW(vector<Neuron> &nextLayerNeurons, Neuron *ghostNeuronTop, Neuron * ghostNeuronBottom);
};

// Private Methods

double Neuron::sigmoid(double x) {
   double expVal = exp(-x);
   return 1.0 / (1.0 + expVal);
}

double Neuron::sigmoidDerivative(double x) {
   return sigmoid(x) * (1 - sigmoid(x));
}

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
   output = 0.0;  // Default neuron value is set to 0.0
   eta = 0.5;
   for (int connection = 0; connection < numOutputs; connection++) {
      Connection c = Connection();
      c.weight = 0.1; // Change this to be a random value
      outputWeights.push_back(c);
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

void Neuron::setGradient(double value) {
   gradient = value;
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


void Neuron::feedForward(vector<Neuron> &prevLayerNeurons, int layerIndex, Neuron *ghostNeuronTop, Neuron *ghostNeuronBottom) {
   double sum = 0.0;
   for (int i = 0; i < prevLayerNeurons.size(); i++) {
      sum += prevLayerNeurons[i].getOutput() *
             prevLayerNeurons[i].getOutputWeights()[index].weight;
   }

   if (layerIndex > 1) {
      // cout << "Rank: " << myRank << " LayerIndex: " << layerIndex << " " << "Neuron: " << index << " Connection from ghostNeuronTop: "
      // << ghostNeuronTop->getOutputWeights().size() << " weight for connection " << ghostNeuronTop->getOutputWeights()[index].weight << endl;
      sum += ghostNeuronTop->getOutput() * ghostNeuronTop->getOutputWeights()[index].weight;
      sum += ghostNeuronBottom->getOutput() * ghostNeuronBottom->getOutputWeights()[index].weight;
   }

   // Use the sigmoid activation function
   if (layerIndex < 2) {
      output = sigmoid(sum);
   } else {
      output = sum;
   }


   // Store in local data so that MPI_Allgather() can access it
   // TODO: Change this so that the last layer is not hard coded to 2
   if (layerIndex == 2) {
      localData[index] = output;
   }

}

double Neuron::sumDOW(vector<Neuron> &nextLayerNeurons, Neuron *ghostNeuronTop, Neuron *ghostNeuronBottom) {
   double sum = 0.0;

   // Sum the contribution of errors at the node that are feedForward
   for (int i = 0; i < nextLayerNeurons.size(); i++) {
      sum += outputWeights[i].weight * nextLayerNeurons[i].gradient;
   }
   // Sum up ghosh neuron errors
   for (int i = 0; i < nextLayerNeurons.size(); i++) {
      sum += ghostNeuronTop->getOutputWeights()[i].weight * nextLayerNeurons[i].gradient;
      sum += ghostNeuronBottom->getOutputWeights()[i].weight * nextLayerNeurons[i].gradient;
   }

   return sum;
}

void Neuron::calcHiddenGradients(vector<Neuron> &nextLayerNeurons, Neuron *ghostNeuronTop, Neuron *ghostNeuronBottom) {
   double dow = sumDOW(nextLayerNeurons, ghostNeuronTop, ghostNeuronBottom);
   // cout << dow << endl;
   gradient = dow * sigmoidDerivative(output);
   // cout << gradient << endl;
}

void Neuron::updateWeights(vector<Neuron> &prevLayerNeurons, Neuron *ghostNeuronTop, Neuron *ghostNeuronBottom) {
   for (int i = 0; i < prevLayerNeurons.size(); i++) {
      Neuron &neuron = prevLayerNeurons[i];
      double oldDeltaWeight = neuron.outputWeights[index].deltaWeight;
      double newDeltaWeight = eta * neuron.getOutput() * gradient + oldDeltaWeight;

      // printf("oldDeltaWeight %f, newDeltaWeight %f\n", oldDeltaWeight, newDeltaWeight);

      neuron.outputWeights[index].deltaWeight = newDeltaWeight;
      neuron.outputWeights[index].weight += newDeltaWeight;
   }

}
