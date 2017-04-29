// Implementation of a fully connected Neural Net (work in progress)
// Mukul Surajiwale

#include <iostream>
#include <vector>
#include <cstdlib>
#include <assert.h>
#include <cmath>

using namespace std;

struct Connection {
   double weight;
   double deltaWeight;
};

// ------------------ Neuron ------------------
class Neuron {
public:
   Neuron(int num_output, int _my_index);
   void setOutputValue(double value) { output_value = value; }
   double getOutputValue() const { return output_value; }
   void feedForward(const vector<Neuron> &prev_layer);
   void calculateOutputGradients(double target_value);
   void calculateHiddenGradients(const vector<Neuron> &next_layer);
   void updateInputWeights(vector<Neuron> &prev_layer);
private:
   static double transferFunction(double x);
   static double transferFunctionDerivative(double x);
   static double randomWeight() { return rand() / double(RAND_MAX); }
   double sumDOW(const vector<Neuron> & next_layer);
   int my_index;
   double output_value;
   double gradient;
   double eta;
   double alpha;
   vector<Connection> output_weights;

};

Neuron::Neuron(int num_output, int _my_index) {
   eta = 0.5;
   alpha = 0.5;
   for (int c = 0; c < num_output; c++) {
      output_weights.push_back(Connection());
      output_weights[c].weight = randomWeight();
   }
   my_index = _my_index;
}

double Neuron::sumDOW(const vector<Neuron> & next_layer) {
   double sum = 0.0;

   // Sum the contribution of errors at the node that are feedForward
   for (int i = 0; i < next_layer.size() - 1; i++) {
      sum += output_weights[i].weight * next_layer[i].gradient;
   }

   return sum;
}

double Neuron::transferFunction(double x) {
   // tanh - output range [-1.0, 1.0]
   return tanh(x);
}

double Neuron::transferFunctionDerivative(double x) {
   // tanh derivative
   return 1 - x * x;
}

void Neuron::calculateOutputGradients(double target_value) {
   double delta = target_value - output_value;
   gradient = delta * Neuron::transferFunctionDerivative(output_value);
}

void Neuron::calculateHiddenGradients(const vector<Neuron> &next_layer) {
   double dow = sumDOW(next_layer);
   gradient = dow * Neuron::transferFunctionDerivative(output_value);
}

void Neuron::updateInputWeights(vector<Neuron> &prev_layer) {
   // weights in the connection container in the previous layer need to be updated

   for (int i = 0; i < prev_layer.size(); i++) {
      Neuron &neuron = prev_layer[i];
      double old_delta_weight = neuron.output_weights[my_index].deltaWeight;
      double new_delta_weight = eta * neuron.getOutputValue() * gradient + alpha + old_delta_weight;

      neuron.output_weights[my_index].deltaWeight = new_delta_weight;
      neuron.output_weights[my_index].weight += new_delta_weight;
   }
}

void Neuron::feedForward(const vector<Neuron> &prev_layer) {
   double sum = 0.0;
   for (int i = 0; i < prev_layer.size(); i++) {
      cout << i << endl;
      sum += prev_layer[i].getOutputValue();
   }

   // Transfer Function
   output_value = Neuron::transferFunction(sum);
}

// ------------------ Net ------------------
class Net {
public:
   Net(const vector<int> &topology);
   void feedForward(const vector<double> &input_Values);
   void backProp(const vector<double> &target_Values);
   void getResults(vector<double> &result_Values) const;
private:
   vector<vector<Neuron> > layers;  // layers[layerNum][neuronNum]
   double error;
   double recentAverageError;
   double recentAverageSmoothingFactor;
};

Net::Net(const vector<int> &topology) {
   int num_layers = topology.size();
   // Add layers from topology
   for (int layer_num = 0; layer_num < num_layers; layer_num++) {
      layers.push_back(vector<Neuron>());
      int num_output = layer_num == topology.size() - 1 ? 0 : topology[layer_num + 1];
      // Add the correct number of neurons to each layer
      // use <= to add one bias neuron to each layer
      for (int neuron_num = 0; neuron_num <= topology[layer_num]; neuron_num++) {
         layers[layer_num].push_back(Neuron(num_output, neuron_num));
      }
   }

   // Force bias nodes output value to be 1.0
   layers.back().back().setOutputValue(1.0);
}

void Net::feedForward(const vector<double> &input_Values) {
   assert(input_Values.size() == layers[0].size() - 1);

   // Feed input values in to the input neurons
   for (int i = 0; i < input_Values.size(); i++) {
      layers[0][i].setOutputValue(input_Values[i]);
   }

   // Forward propogate
   for (int layer_num = 1; layer_num < layers.size(); layer_num++) {
      vector<Neuron> &prev_layer = layers[layer_num - 1];
      cout << "layer num " << layer_num << endl;
      for (int n = 0; n < layers[layer_num].size(); n++) {
         cout << n << endl;
         layers[layer_num][n].feedForward(prev_layer);
      }
   }

}

void Net::backProp(const vector<double> &target_Values) {
   // Calculate overall net error

   vector<Neuron> &output_layer = layers.back();

   for (int i = 0; i < output_layer.size(); i++) {

      double delta = target_Values[i] - output_layer[i].getOutputValue();
      error += delta * delta;
   }
   error /= output_layer.size() - 1;  // average error squared
   error = sqrt(error);  // RMS

   // Get error stastics
   recentAverageError = (recentAverageError * recentAverageSmoothingFactor + error) / (recentAverageSmoothingFactor + 1.0);

   // Calculate ouput gradients
   for (int i = 0; i < output_layer.size(); i++) {
      output_layer[i].calculateOutputGradients(target_Values[i]);
   }

   // Calculate gradients on hidden layers
   for (int layer_num = layers.size() - 2; layer_num > 0; layer_num--) {
      vector<Neuron> &hidden_layer = layers[layer_num];
      vector<Neuron> &next_layer = layers[layer_num + 1];

      for (int i = 0; i < hidden_layer.size(); i++) {
         hidden_layer[i].calculateHiddenGradients(next_layer);
      }
   }

   // Update connection weights
   for (int layer_num = layers.size() - 1; layer_num > 0; layer_num--) {
      vector<Neuron> &layer = layers[layer_num];
      vector<Neuron> &prev_layer = layers[layer_num - 1];

      for (int i = 0; i < layers.size(); i++) {
         layer[i].updateInputWeights(prev_layer);
      }
   }
}

void Net::getResults(vector<double> &result_Values) const {
   result_Values.clear();
   for (int i = 0; i < layers.back().size() - 1; i++) {
      result_Values.push_back(layers.back()[i].getOutputValue());
   }
}

int main() {

   vector<int> topology;
   // Construct a net with 1 input layer, 1 hidden layer, and 1 output layer
   topology.push_back(3);  // 3 input neurons
   topology.push_back(2);  // 2 hidden neurons
   topology.push_back(1);  // 1 output neuron
   Net myNet(topology);

   vector<double> input_Values;
   input_Values.push_back(0.0);
   input_Values.push_back(1.0);
   input_Values.push_back(0.0);

   vector<double> target_Values;
   target_Values.push_back(0.0);

   vector<double> input_Values2;
   input_Values2.push_back(0.0);
   input_Values2.push_back(1.0);
   input_Values2.push_back(1.0);

   vector<double> target_Values2;
   target_Values2.push_back(1.0);




   vector<double> resultValues;

   myNet.feedForward(input_Values);
   myNet.backProp(target_Values);

   myNet.feedForward(input_Values2);
   // myNet.backProp(target_Values2);
   // myNet.getResults(resultValues);
}
