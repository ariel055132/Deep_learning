## Part 2 The mathematical building blocks of neural networks

We will learn

1. A first example of neural network  (I cover this in 2_1.md)
2. Tensors and tensor operations, differentiation
3. How neural networks learn via backpropagation and gradient descent

*2.2 Data representations for neural networks*

All current machine-learning systems use tensors as their basic data structure. (a container for data--almost always numerical data)

*2.2.1 Scalars (0D tensors)*

A tensor that contains only one number is called a scalar (aka scalar tensor, 0D tensor).

In Numpy, a float32 or float64 number is a scalar tensor.

ndim: display the number of axes of a Numpy tensor; a scalar tensor has 0 axes

The no of axes of a tensor is also called its rank.

*2.2.2 Vectors (1D tensors)*

A array of numbers is called a vector (aka 1D tensor), have exactly one axis.8

*2.2.3 Matrices(2D tensors)*

An array of vectors. It has two axes (referred to rows and columns)

*2.2.5 Key attributes of tensors*

1. Number of axes (Rank)
2. Shape: a tuple of integers that describes how many dimensions the tensor has along each axis
3. Data type (dtype in Python library)

*2.2.8 Real-world examples of data tensors*

1. Vector Data: 2D tensors of shape (samples, features)
2. Timeseries data or sequence data: 3D tensors of shape (samples, timesteps, features)
3. Images: 4D tensors of shape (samples, height, width, channels / samples, channels, height, width)
4. Video: 5D tensors of shape (samples, frames, height, width, channels / samples, frames, channels, height, width) 

*2.3 Tensor Operations*

binary operations (AND, OR, NOR, etc)

1. Element-like operation: each element do calculation independently (add, sub, mul is '*')
2. Broadcasting: for calculations between arrays with different shape, the arrays must implement element-like operation
3. Tensor dot (aka tensor product): dot('.') similar to matrix multiplication
4. Reshaping: rearrange its rows and columns to match a target shape

*2.4.1 Derivative*

Imagine as the slope, derived result is the gradient

Differentiable: "can be derived" (e.g: smooth, continuous functions can be derived)

*2.4.3 Stochastic (Random) gradient descent*

1. Find an random position as starting point.
2. Calculate the gradient.
3. Go to the point which has negative gradient until it go to minimum. (The slope not change anymore)