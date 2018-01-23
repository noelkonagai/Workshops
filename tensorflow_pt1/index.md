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



