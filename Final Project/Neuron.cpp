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
   void feedForward(vector<Neuron*> &prevLayerNeurons, int layerIndex, Neuron *ghostNeuronTop, Neuron *ghostNeuronBottom);
   void calcHiddenGradients(vector<Neuron*> &nextLayerNeurons, Neuron *ghostNeuronTop, Neuron *ghostNeuronBottom);
   void updateWeights(vector<Neuron*> &prevLayerNeurons, Neuron *ghostNeuronTop, Neuron *ghostNeuronBottom, int layerNum);
   double sumOutputValues(vector<Neuron*> &nextLayerNeurons, Neuron *ghostNeuronTop, Neuron * ghostNeuronBottom);
   void setOutputWeightForIndex(int index, double _weight);
   void setOutputDeltaWeightForIndex(int index, double _dweight);
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
   eta = 0.001;  // Default learning rate
   for (int connection = 0; connection < numOutputs; connection++) {
      Connection c = Connection();
      c.weight = 0.1;
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

void Neuron::setOutputWeightForIndex(int index, double _weight) {
   outputWeights[index].weight = _weight;
}

void Neuron::setOutputDeltaWeightForIndex(int index, double _dweight) {
   outputWeights[index].deltaWeight = _dweight;
}


void Neuron::feedForward(vector<Neuron*> &prevLayerNeurons, int layerIndex, Neuron *ghostNeuronTop, Neuron *ghostNeuronBottom) {
   double sum = 0.0;
   for (int i = 0; i < prevLayerNeurons.size(); i++) {
      sum += prevLayerNeurons[i]->getOutput() *
             prevLayerNeurons[i]->getOutputWeights()[index].weight;
   }

   if (layerIndex > 1) {
      sum += ghostNeuronTop->getOutput() * ghostNeuronTop->getOutputWeights()[index].weight;
      sum += ghostNeuronBottom->getOutput() * ghostNeuronBottom->getOutputWeights()[index].weight;
   }

   // Use the sigmoid activation function
   if (layerIndex < 2) {
      output = sigmoid(sum);
   } else {
      output = sum;
   }

   // if (layerIndex == 4) {
   //    localData[index] = output;
   // }

}

double Neuron::sumOutputValues(vector<Neuron*> &nextLayerNeurons, Neuron *ghostNeuronTop, Neuron *ghostNeuronBottom) {
   double sum = 0.0;

   // Sum the contribution of errors at the node that are feedForward
   for (int i = 0; i < nextLayerNeurons.size(); i++) {
      sum += outputWeights[i].weight * nextLayerNeurons[i]->gradient;
   }
   // Sum up ghost neuron errors
   for (int i = 0; i < nextLayerNeurons.size(); i++) {
      sum += ghostNeuronTop->getOutputWeights()[i].weight * nextLayerNeurons[i]->gradient;
      sum += ghostNeuronBottom->getOutputWeights()[i].weight * nextLayerNeurons[i]->gradient;
   }

   return sum;
}

void Neuron::calcHiddenGradients(vector<Neuron*> &nextLayerNeurons, Neuron *ghostNeuronTop, Neuron *ghostNeuronBottom) {
   double dow = sumOutputValues(nextLayerNeurons, ghostNeuronTop, ghostNeuronBottom);
   gradient = dow * sigmoidDerivative(output);
}

void Neuron::updateWeights(vector<Neuron*> &prevLayerNeurons, Neuron *ghostNeuronTop, Neuron *ghostNeuronBottom, int layerNum) {
   for (int i = 0; i < prevLayerNeurons.size(); i++) {
      Neuron *neuron = prevLayerNeurons[i];
      double oldDeltaWeight = neuron->outputWeights[index].deltaWeight;
      double newDeltaWeight = eta * neuron->getOutput() * gradient + oldDeltaWeight;

      neuron->setOutputDeltaWeightForIndex(index, newDeltaWeight);

      double oldW = neuron->outputWeights[index].weight;
      double newW = oldW + newDeltaWeight;

      neuron->setOutputWeightForIndex(index, newW);

      double current = neuron->getOutputWeights()[index].weight;
      neuron->outputWeights[index].deltaWeight = newDeltaWeight;
      neuron->outputWeights[index].weight += newDeltaWeight;

   }

}
