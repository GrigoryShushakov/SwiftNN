import AppKit
import Accelerate

//MARK:Funcs
//Neuron activation function
func sigmoidFunction(_ x: Double) -> Double {
    return 1.0 / (1.0 + exp(-x))
}
//For calc back propagation delta weights
func derivativeSigmoidFunction(_ x: Double) -> Double {
    return sigmoidFunction(x) * (1 - sigmoidFunction(x))
}
//Initial neuron weights
func randomWeights(number: Int) -> [Double] {
    return (0..<number).map{ _ in Double(arc4random()) / Double(UInt32.max)}
}

//MARK:Neuron
class Neuron {
    var weights: [Double]
    var activationFunction: (Double) -> Double
    var derivativeActivationFunction: (Double) -> Double
    var delta: Double = 0.0
    var inputCache: Double = 0.0
    var learningRate: Double
    init(weights: [Double],
         activationFunction: @escaping (Double) -> Double,
         derivativeActivationFunction: @escaping (Double) -> Double,
         learningRate: Double) {

        self.weights = weights
        self.activationFunction = activationFunction
        self.derivativeActivationFunction = derivativeActivationFunction
        self.learningRate = learningRate
    }
}

extension Neuron {
    func neuronOutput(inputs: [Double]) -> Double {
        inputCache = zip(inputs, self.weights).map(*).reduce(0, +)
        return activationFunction(inputCache)
    }
}

//MARK:Layer
class Layer {
    let previousLayer: Layer?
    var neurons: [Neuron]
    var layerOutputCache: [Double]

    init(previousLayer: Layer? = nil,
         numberOfNeurons: Int,
         activationFunction: @escaping (Double) -> Double,
         derivativeActivationFunction: @escaping (Double)-> Double,
         learningRate: Double) {

        self.previousLayer = previousLayer
        self.neurons = Array<Neuron>()
        for _ in 0..<numberOfNeurons {
            self.neurons.append (Neuron(weights: randomWeights(number: previousLayer?.neurons.count ?? 0),
                                        activationFunction: activationFunction,
                                        derivativeActivationFunction: derivativeActivationFunction,
                                        learningRate: learningRate))
        }
        self.layerOutputCache = Array<Double>(repeating: 0.0,
                                         count: neurons.count)
    }

    //Forward propagation prediction outputs calc
    func outputSinapses(inputs: [Double]) -> [Double] {
        if previousLayer == nil { //Input layer
            layerOutputCache = inputs
        } else { //Hidden and output layers
            layerOutputCache = neurons.map { $0.neuronOutput(inputs: inputs) }
        }
        return layerOutputCache
    }

    //Backward propagation deltas calc
    func calculateDeltasForOutputLayer(expected: [Double]) {
        for n in 0..<neurons.count {
            neurons[n].delta = neurons[n].derivativeActivationFunction( neurons[n].inputCache) * (expected[n] - layerOutputCache[n])
        }
    }

    //Backward propagation deltas calc
    func calculateDeltasForHiddenLayer(nextLayer: Layer) {
        for (index, neuron) in neurons.enumerated() {
            let nextWeights = nextLayer.neurons.map { $0.weights[index] }
            let nextDeltas = nextLayer.neurons.map { $0.delta }
            let sumOfWeightsXDeltas = zip(nextWeights, nextDeltas).map(*).reduce(0, +)
            neuron.delta = neuron.derivativeActivationFunction( neuron.inputCache) * sumOfWeightsXDeltas
        }
    }
}

//MARK:Network
class Network {
    var layers: [Layer]

    init(layerStructure:[Int],
         activationFunction: @escaping (Double) -> Double = sigmoidFunction,
         derivativeActivationFunction: @escaping (Double) -> Double = derivativeSigmoidFunction,
         learningRate: Double) {

        if (layerStructure.count < 3) {
            print("Error: Should be at least 3 layers (1 input, 1 hidden, 1 output)")
        }
        layers = [Layer]()

        //Create input layer
        layers.append (Layer(numberOfNeurons: layerStructure[0],
                            activationFunction: activationFunction,
                            derivativeActivationFunction: derivativeActivationFunction,
                            learningRate: learningRate))

        //Create hidden layers and output layer
        for layer in layerStructure.enumerated() where layer.offset != 0 {
            layers.append (Layer(previousLayer: layers[layer.offset - 1],
                                 numberOfNeurons: layer.element,
                                 activationFunction: activationFunction,
                                 derivativeActivationFunction: derivativeActivationFunction,
                                 learningRate: learningRate))
        }
    }

    //Forward propagation prediction
    func outputs(input: [Double]) -> [Double] {
        return layers.reduce(input) { $1.outputSinapses(inputs: $0) }
    }

    //Backward propagation training, calc deltas
    func backwardPropagationMethod(expected: [Double]) {
        layers.last?.calculateDeltasForOutputLayer(expected: expected)
        for l in 1..<layers.count - 1 {
            layers[l].calculateDeltasForHiddenLayer(nextLayer: layers[l + 1])
        }
    }

    //Apply new weights for neurons after each learning epoch
    func updateWeightsAfterLearn() {
        for layer in layers {
            for neuron in layer.neurons {
                for w in 0..<neuron.weights.count {
                    neuron.weights[w] = neuron.weights[w] + (neuron.learningRate * (layer.previousLayer?.layerOutputCache[w])!  * neuron.delta)
                }
            }
        }
    }

    //Training network epoch
    func train(inputs:[[Double]], expecteds:[[Double]]) {
        for (position, input) in inputs.enumerated() {
            let expectedOutputs = expecteds[position]
            let currentOutputs = outputs(input: input)
            let diffrencesBetweenPredictionAndExpected = zip(currentOutputs, expectedOutputs).map{$0-$1}
            let meanSquaredError = sqrt(diffrencesBetweenPredictionAndExpected.map{$0*$0}.reduce(0,+))
            print("Training loss: \(meanSquaredError)")

            backwardPropagationMethod(expected: expectedOutputs)
            updateWeightsAfterLearn()
        }
    }

    //Validation results
    func validate(input:[Double], expected:Double) -> (result: Double, expected:Double) {
        let result = outputs(input: input)[0]
        return (result,expected)
    }
}

var network:Network = Network(layerStructure: [3,2,1], learningRate: 0.4)

let trainEpochs = 500

let trainingPatterns = [[0.0,0.0,0.0],
                        [0.0,0.0,1.0],
                        [0.0,1.0,0.0],
                        [0.0,1.0,1.0],
                        [1.0,0.0,0.0],
                        [1.0,0.0,1.0],
                        [1.0,1.0,0.0],
                        [1.0,1.0,1.0]]

let expectedResults = [[0.0],
                       [1.0],
                       [0.0],
                       [0.0],
                       [1.0],
                       [1.0],
                       [0.0],
                       [1.0]]

for epoch in 0..<trainEpochs {
    network.train(inputs: trainingPatterns, expecteds: expectedResults)
}

print ("\nWeights of hidden layer")
for neuron in network.layers[1].neurons {
    print("\(neuron.weights)")
}

print ("\nWeights of output layer")
for neuron in network.layers[2].neurons {
    print("\(neuron.weights)\n")
}

for i in 0..<trainingPatterns.count {
    let results = network.validate(input: trainingPatterns[i], expected: expectedResults[i][0])
    print("For input:\(trainingPatterns[i]) the prediction is:\(results.result), expected:\(results.expected)")
}

