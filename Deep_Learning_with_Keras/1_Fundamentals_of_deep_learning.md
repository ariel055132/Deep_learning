## Part 1 Fundamentals of Deep Learning

*1.1 Artificial intelligence, machine learning and deep learning*

Artificial intelligence (largest set) --> machine learning --> deep learning (smallest subset)

*1.1.2 Machine Learning*

In Classical programming, we input rules and data, the program will output answers.

In machine learning, we input data and answers, the machine will output rules. Machine learning tends to deal with large, complex datasets and little mathematical theory.

*1.1.3 Learning representations from data*

We need three things in machine-learning.

1. Input data points.
2. Examples of expected output
3. A way to measure whether 

A machine learning model transforms its input data into meaningful outputs by finding appropriate representations for their input data -- transformations of the data that make it more amenable to the task at hand.

The example 1.3 can be solved by single perceptron.

So that's what machine learning is, technically: searching for useful representations of some input data, within a predefined space of possibilities, using guidance from a feedback signal. This simple idea allows for solving a remarkably broad range of intellectual tasks.

*1.1.4 The "deep" in deep learning* 

It is a specific subfield of machine learning, a new take on learning representations from data that puts an emphasis on learning successive layers of increasingly meaningful representations.

"deep" stands for the idea of successive layers of representations.

Neural network is always used in deep learning.

Deep learning is a mathematical frame-work for learning representations from data.

*1.1.5 Understanding how deep learning works, in three figures*

Machine learning: mapping inputs to targets, which is done by observing many examples of inputs and targets.

Deep neural networks: input-to-target mapping via a deep sequence of simple data transformations (layers) and that these transformations are learned by exposure to examples.

The specifications of what a layer does to its input data is stored in the layer's weights, which in essence are a bunch of numbers. 

The transformation implemented by a layer is parameterized by its weights (Goal: Finding the right values for these weights of all layers in a network, such that the network will correct map example inputs to their associated targets.)

Loss functions: Measure how far this output is from what you expected. Takes the predictions of the network and the true target (what you want the network to output) and computes a distance score, capturing how well the network has done on this specific example.

Optimizer: Use the output from loss functions as feedback signal to adjust the value of the weight. (aka Backpropagation algorithm)

Training loop/epoch: The weight will be adjusted a little in the correct directions and loss score decreases. It will repeat number of times.

*1.2.1 Probabilistic modeling*

It is the application of the principles of statistics to data analysis. Incorporate random variables and probability distributions into the model of an event. 

1. Naive Bayes algorithm: a type of ML classifier based on applying Bayes' theorem while assuming that the features in the input data are all independent.
2. Logistic regression.

*1.2.3 Kernel method*

a group of classification algorithms

1. SVM (Support vector machine): solving classification problems by finding good decision boundaries between two set of points belonging to two different categories.

*1.2.4 Decision trees, random forests, gradient boosting machines*

Decision trees: flowchart-like structures that let you classify input data points or predict output values given inputs. Parameters that are learned are the questions about the data.

Random forest: A robust, practical take on decision-tree learning that involves building a large number of specialized decision trees and then ensembling their outputs.

Gradient boosting machine: A machine-learning technique based on ensembling weak prediction models

*1.3 why deep learning*

Different ideas of deep learning are developed.

1. Convolutional neural networks and backpropagation (computer vision)
2. Long Short-Term Memory (LSTM) algorithm (Deep learning)

Three technical forces are more advanced

1. Hardware.
2. Datasets and benchmarks
3. Algorithm
   1. Better activation functions for neural layers
   2. Better weight-initialization schemes
   3. Better optimization schemes (optimizer)