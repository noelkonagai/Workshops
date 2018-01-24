# TensorFlow: The Basics

[Home Page](https://noelkonagai.github.io/Workshops/)

TensorFlow was originally developed by researchers and engineers working on the Google Brain Team within Google's Machine Intelligence research organization for the purposes of conducting machine learning and deep neural networks research, but the system is general enough to be applicable in a wide variety of other domains as well. [TensorFlow.org](https://www.tensorflow.org/)

In this workshop you will become familiar with the building blocks of Tensorflow models, such as tensors, placeholders, variables, graphs, and sessions. At the end of the workshop, you will be able to create a simple regression model.

In order to install TensorFlow on your own device, please consult this [tutorial](https://www.tensorflow.org/install/).

## Getting started: Constants, Graphs, Sessions

First and foremost import these two libraries into your python program.

```python
import tensorflow as tf
```

And check what version of TensorFlow you have installed on your computer. This workshop uses 1.4.0

```python
print(tf.__version__)
```
```bash
1.4.0
```

So let's see an example of a TensorFlow computation graph. A computational graph is a series of TensorFlow operations arranged into a graph of nodes. Think of graph as a sequence of operations, nodes being the operations themselves.

```python
tensor_1 = tf.constant(3.0, dtype = tf.float32)
tensor_2 = tf.constant(4.0) #the datatype of float32 is inferred in this case
```

We can use ```tf.add``` API to add these two tensors. Try the following.

```python
tensor_3 = tf.add(tensor_1, tensor_2)
print(tensor_3)
```

This will print the following.

```bash
Tensor("Add:0", shape=(), dtype=int32)
```

You might have expected it to print 7.0. However, each of these operations (nodes) need to be evaluated in order to display the actual value. In order to do so, we need to create a TensorFlow session. 

```python
sess = tf.Session()
print(sess.run(tensor_3))
```

```bash
7.0
```

**Now you might ask, why do we need ```tf.Session()```?**

Well, there are a number of reasons, one being this way we can create multiple graphs, and run different graphs each time we call ```sess.run()```. Another being, we can set session configurations, such as GPU options. You can read up on graphs and session via this [official documentation](https://www.tensorflow.org/programmers_guide/graphs).

## Using Placeholders

Quite frequently you will be using ```tf.placeholder``` instead of ```tf.constant``` because you will be creating a machine learning model that you will use for training, testing, and evaluating. The data you will feed into the graph should be dynamic and in order to achieve that, we have a handy API ```tf.placeholder```.

The previous code can be rewritten as

```python
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

adder_node = a + b
```

Here you defined two placeholders that will accept float32 values and an adder node, ```+``` is a shortcut for ```tf.add()```. Since we have a session running we can write the following.

```python
sess.run(adder_node, {a: 3.0, b: 4.0})
```
```bash
7.0
```

The second argument after ```adder_node``` is a feed dictionary. Try now creating more complex graphs. Hint, you can sequence your operations by making a list as follows.

```python
op_1 = a + b
op_2 = a ** b
op_3 = a / b

operations = [op_1, op_2, op_3]
```

Now, you can also run an interactive session, where with the ```.eval()``` API call you can evaluate without having to run the session every time.

```python
sess = tf.InteractiveSession() #run it only once

for op in operations:
	print(op.eval())
	print("\n")
```

If you are curious what is the difference between ```tf.Session()``` and ```tf.InteractiveSession()``` read [this Stackoverflow post](https://stackoverflow.com/questions/41791469/whats-the-difference-between-tf-session-and-tf-interactivesession).

## Recap of Matrix Multiplication

Since one of the building blocks of machine learning is a thorough understanding of linear algebra, let us recap matrix multiplication with ```tf.matmul()```.

Create two tensor constants with shape ```[3,3]``` and ```[3,2]```

```python
a = tf.constant([1, 2, 3, 4, 5, 6, 7, 8, 9], shape = [3, 3])
b = tf.constant([6, 5, 4, 3, 2, 1], shape = [3, 2])
```

and now multiply them.

```python
c = tf.matmul(a,b)
c.eval()
```
```bash
array([[20, 14],
       [56, 41],
       [92, 68]], dtype=int32)
```

**We keep talking about tensors, but what on earth are they?**

> The central unit of data in TensorFlow is the tensor. A tensor consists of a set of primitive values shaped into an array of any number of dimensions. A tensor's rank is its number of dimensions. ([source](https://www.tensorflow.org/get_started/get_started))

```python
3 # a rank 0 tensor; a scalar with shape []
[1., 2., 3.] # a rank 1 tensor; a vector with shape [3]
[[1., 2., 3.], [4., 5., 6.]] # a rank 2 tensor; a matrix with shape [2, 3]
[[[1., 2., 3.]], [[7., 8., 9.]]] # a rank 3 tensor with shape [2, 1, 3]
```

More information on the ```tf.matmul()``` API [here](https://www.tensorflow.org/api_docs/python/tf/matmul).

## Simple Regression Task

**Open Jupyter notebook and create a new file**

Use the commands below to import the necessary libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
%matplotlib inline
```

And let's create an artificial dataset. Let's make an ```x_data``` array of random numbers between 0 and 10. Let's make a ```y_true``` value array by adding a random ```noise``` term to ```x_data + b```, where ```b``` is the intercept. This way we get a slightly noisy, more realistic dataset.

```python
x_data = np.linspace(0.0, 10.0, 1000000)
noise = np.random.randn(len(x_data))
b = 5

y_true = (0.5 * x_data ) + b + noise

# and now create a pandas dataframe out of it
my_data = pd.concat([pd.DataFrame(data=x_data,columns=['x']),pd.DataFrame(data=y_true,columns=['y'])],axis=1)
```
Plotting it out you will get something like this.
```python
my_data.sample(n=500).plot(kind='scatter',x='x',y='y')
```

![Figure 1](/figures/figure_1.png)

Let us set up the building blocks of our simplre regression model.

First let's initialize the variables m (slope) and b (y-intercept) at some point. This is what our Machine Learning model will adjust through learning from the loss. 
 
```python
m = tf.Variable(0.5)
b = tf.Variable(1.0)
```

As discussed earlier, we will need placeholders to which we will feed in our data. We need to tell what's the size of this placeholder. **Note to self: explain the concept of batch_size**

```python
batch_size = 10 

x_placeholder = tf.Placeholder(tf.float32, [batch_size])
y_placeholder = tf.Placeholder(tf.float32, [batch_size])
```

Then create our graph, which a simple linear regression with one variable.

```python
y_model = m * x_placeholder + b
```

And set up a loss function from which our model will learn to adjust. We use mean squared error (MSE) as the error metrics. It is calculated by adding the squares of the differences of the predicted values (y_model) and the actual y values (y_placeholder). In mathematical notation, it MSE is **Note to self: add MSE**

```python
error = tf.reduce_sum( tf.square( y_placeholder - y_model ))
```

(Workshops materials to be continued and updated.)


