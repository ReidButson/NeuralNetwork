import neural_functions as nf
import numpy as np


def table_row(values, space=10):
    row = '|'.join(str(val).ljust(space) for val in values)
    return '|' + row + '|'


class NeuralNet:

    def __init__(self, inputs, outputs, hidden=None):

        self.synapses = []

        self.weights = []
        self.delta_weights = []

        self.thresholds = []
        self.delta_thresholds = []

        self.output = []
        self.error = 1

        self.sum_squares = []

        if not hidden:
            self.weights.append(np.random.rand(inputs, outputs))

        else:
            self.weights.append(np.random.rand(inputs, hidden[0]))
            self.thresholds.append(np.random.rand(hidden[0]))

            for i in range(0, len(hidden)-1):
                self.weights.append(np.random.rand(hidden[i], hidden[i+1]))
                self.thresholds.append(np.random.rand(hidden[i+1]))

            self.weights.append(np.random.rand(hidden[-1], outputs))
            self.thresholds.append(np.random.rand(outputs))
            pass

        self.delta_weights = [0]*len(self.weights)
        self.delta_thresholds = [0]*len(self.thresholds)

    def activation(self, data):
        self.synapses = []
        synapse = None

        for i, (weight, threshold) in enumerate(zip(self.weights, self.thresholds)):
            if i == 0:
                synapse = nf.synapse(data, weight, threshold)

            else:
                synapse = nf.synapse(synapse, weight, threshold)

            self.synapses.append(synapse)

        return synapse

    def weight_training(self, data, desired, learn_rate):
        gradient = None

        for i in range(1, len(self.weights) + 1):
            if i == 1:
                gradient = nf.gradient_output(desired, self.synapses[-i])
                self.error = np.sum([x**2 for x in gradient])
            else:
                gradient = nf.gradient_hidden(self.synapses[-i], gradient, self.weights[-(i-1)])

            if i == len(self.weights):
                self.delta_weights[-i] = nf.weight_delta(learn_rate, data, gradient)

            else:
                self.delta_weights[-i] = nf.weight_delta(learn_rate, self.synapses[-(i + 1)], gradient)

            self.delta_thresholds[-i] = nf.threshold_delta(learn_rate, gradient)

    def adjust_weights(self):
        for i in range(len(self.weights)):
            self.weights[i] = np.array(self.weights[i]) + np.array(self.delta_weights[i])
            self.thresholds[i] = np.array(self.thresholds[i]) + np.array(self.delta_thresholds[i])

    def train(self, data, expected_output, accepted_error, to_console=False):

        err = 1
        epoch = 0
        while err > accepted_error:
            if to_console:
                print(table_row(['Epoch {}'.format(epoch), 'Input', 'Expected Value', 'Output', 'error'], 25))
                print(('|' + '-' * 25) * 5 + '|')
            err = 0
            for d, e in zip(data, expected_output):
                err = self.error
                out = self.activation(d)
                if to_console:
                    print(table_row(['', d, e, out, self.error], 25))
                #print('E:  ', e)
                #print('O:  ', out)
                self.weight_training(d, e, 0.1)
                self.adjust_weights()
            if to_console:
                print(('|' + '-' * 25) * 5 + '|')
            self.sum_squares.append(err)
            epoch += 1
