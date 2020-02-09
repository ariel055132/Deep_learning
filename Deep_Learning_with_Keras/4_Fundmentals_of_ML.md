## Part 4 Fundamentals of machine learning

Consolidate concepts in chapter 3 (model evaluation, data preprocessing and feature engineering) into a detailed seven step workflow for machine learning task.

*Four branches of ML*

1. Supervised learning: Learning to map input data to known objects, given a set of examples. (e.g OCR optical character recognition, speech recognition, image classification, language translation)

2. Unsupervised learning: Finding transformation of the input data without the help of any targets (e.g dimensionality reduction and clustering)

3. Self-supervised learning: a supervised learning without human-annotated labels (e.g autoencoders .... xlnet)

4. Reinforcement learning: an agent receives information and learns to choose actions that will maximize some reward.

   

*Terms definition*

1. Sample / Input : data point that goes into your model.

2. Prediction / Output : what comes out of your model.

3. Target: the truth. What your model should ideally have predicted.

4. Prediction error / loss value: A measure of the distance between your model's prediction and the target.

5. Classes: A set of possible labels to choose from in a classification problem.

6. Label: A specific instances of a class annotation in a classification problem.

7. Ground-truth / annotations: All targets for a dataset, typically collected by humans.

8. Binary classification: A classification task where each input sample should be categorized into two exclusive categories. 

9. Multiclass classification: A classification task where each input sample should be categorized into more than two categories. (e.g classifying handwritten digits)

10. Multilabel classification: A classification task where each input sample can be assigned multiple labels. (e.g a given image may contain a cat and a dog and should be annotated with the 'cat' label and the 'dog' label)

11. Scalar regression: A task where the target is a continuous scalar value. 

12. Vector regression: A task where target is a set of continuous values.

13. Mini-batch / batch: A small set of samples that are processed simultaneously by the model.

    

*4.2 Evaluating machine-learning models*

In machine-learning, the goal is to achieve models that generalize-- that perform well on never-before-seen data.

1. How to measure generalization

2. How to evaluate machine-learning models

   

*4.2.1 Training, validation and test sets*

Why validation sets is need?

Developing a model always involves tuning its config. (number of layers / batch-size of the layers). Use validation set to get the feedback signal the performance of the model and think whether the config need to be changed.

Training sets is used for training the model, if it is used too much, it may need to overfitting. (The model learn how to classify training data, which lowers the generalization of the model)

Three classic evaluation recipes:

1. Simple Hold-out validation: Set apart some fraction of data as test set. Train on the remain data, and evaluate on the test set. In order to prevent information leaks, fine tune model based on validation set. (Disadvantage: If little data is available, the validation and test set contain too few sample to be statistically representative of the data at head. )
2. K-fold validation: split the data into K partition of equal size. For each partition i, train a model on the remaining K-1 partitions. The final score is then the averages of the K scores obtained. This method is helpful when the performance of model shows significant variance based on your train-test split.
3. Iterated K-fold validation with shuffling: Used when you have relatively little data available

​      

*4.3 Data pre-processing, feature engineering and feature learning*

Data pre-processing: making the raw data at hand more amenable to neural networks. (vectorization, normalization, handling missing values, feature extraction)

1. Vectorization: Turn the data into tensors (data vectorization).

2. Value normalization: normalize each feature independently and let it have a standard derivation of 1 and a mean of 0.
   1. Take small values: most values in the 0-1 range.
   2. Be homogeneous (same): all features should take values in roughly the same range.

3. Handling missing values: artificially generate training samples with missing entries, it is safe to input missing as 0 with neural networks.

4. Feature engineering: the process of using your own knowledge about the data and about the ML algorithm at hand to make the algorithm  work better by applying hardcoded transformations to the data before it goes into the model. (making a problem easier by expressing it in a simpler way)

   

*4.4 Overfitting and underfitting*

Optimization: the process of adjusting a model to get the best performance possible on the training data.

Generalization: how well the trained model performs on data it has never seen before.

Underfitting: The model cannot fit the training data. (Maybe the no of parameter is too less) The lower the loss on training data, the lower the loss on test data. 

Overfitting: the model quickly started to overfit to the training data, generalization is lowered.(maybe the no of parameter is too many)

Solution: Regulation

1. Get more training data, a model trained on more data will naturally generalize better. 
2. Modulate the quantity of information that the model is allowed to store (Limit the no of parameter) 
3. Add constraints on what information it is allowed to store.



4.4.1 Reducing the network's size*

The simplest way to prevent overfitting is to reduce the size of model: The number of learnable parameters in the model.

#### Deep learning models tend to be good at fitting to the training data, but the real challenge is generalization, not fitting.

If the network has limited memorization process, it will not be able to learned the mapping as easily.

There is no formula to determine the right number of layers or the right size for each layer.

Evaluate an array of different architectures on your validation set in order to find the correct model size for your data.



*4.4.2 Adding weight regularization*

The principle of Occam's razor: giving two explanations for something, the explanation most likely to be correct is the simplest one, the one that makes fewer assumptions.

Therefore, simpler models are less likely to over-fit than complex ones.

Simpler models: the distribution of parameter values has less entropy / a model with fewer parameters

There is a common way to mitigate overfitting is to put constraints on the complexity of a network by forcing its weights to take only small values which makes the distribution of weight values more regular.

1. L1 regularization: The cost added is proportional to the absolute value of the weight coefficients (the L1 norm of the weights)

2. L2 regularization / Weight decay: The cost added is proportional to the square of the value of the weight coefficients (the L2 norm of the weights) 

   

*4.4.3 Adding dropout*

Dropout, applied to a layer, consists of randomly dropping out a number of output features of the layer during training.



*4.5 The workflow of machine learning*

1. Problem definition
2. Evaluation
3. Feature engineering
4. Fighting overfitting



*4.5.1 Defining the problem and assembling a dataset*

1. What will your input data be?
2. What are you trying to predict?
3. What type of problem are you facing?
4. Binary classification? Multiclass classification? Scalar regression? Vector regression? Multiclass, multilabel classification?



*4.5.2 Choosing a measure of success*

Definition of success? Accuracy, precision and recall, customer-retention rate?

It will lead to the loss function you choose.



*4.5.3 Deciding on an evaluation protocol*

1. Maintain a hold-out validation set
2. Doing K-fold cross-validation: too few samples for validation set
3. Doing iterated K-fold validation: performing highly accurate model evaluation when little data is available



*4.5.4 Preparing the data*

Preparing the data as tensors.



*4.5.5 Developing a model*

1. Last-layer activation: establish constraints on the network's output.

2. Loss function: match the type of problem you are going to solve

3. Optimizer config: use which optimizer / learning rate? Safest: rmsprop

   

Choosing the right activation and loss function:

1. Binary classification: sigmoid, binary_crossentropy
2. Multiclass / single label classification: softmax, categorical_crossentropy
3. Multiclass / Multilabel classification: sigmoid, binary_crossentropy
4. Regression to arbitrary values: None, mse(mean squared error)
5. Regression to values between 0 to 1: sigmoid, mse / binary_crossentropy



*4.5.6 Developing a model that overfits*

1. Add layers
2. Make the layers bigger.
3. Train for more epochs.



*4.5.7 Regularizing the model and tuning parameters*

1. Add dropout
2. Try different architectures / add or remove layers
3. Add L1 and L2 regularization
4. Try different hyperparameters