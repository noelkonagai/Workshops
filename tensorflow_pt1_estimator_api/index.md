# TensorFlow: The Basics -  Estimator API

[Home Page](https://noelkonagai.github.io/Workshops/)


## Creating the data

In order to complete this exercise, please create your artificial data in the following way.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
%matplotlib inline

x_data = np.linspace(0.0, 10.0, 1000000)
noise = np.random.randn(len(x_data))
b = 5

y_true = (0.5 * x_data ) + b + noise

my_data = pd.concat([pd.DataFrame(data=x_data,columns=['x']),pd.DataFrame(data = y_true,columns=['y'])],axis=1)
```

## Using tf.estimator API

The ```tf.estimator.LinearRegressor``` takes in feature columns as its argument. For this you need to create a feature column with ```tf.feature_column.numeric_column```. There are other types of columns as well. You can read more through the official API guide [tf.feature_column](https://www.tensorflow.org/api_docs/python/tf/feature_column).

```python
feat_cols = [tf.feature_column.numeric_column('x',shape=[1])]
estimator = tf.estimator.LinearRegressor(feature_columns=feat_cols)
```

## Train-Test split of your data with sklearn

Machine learning typically involves splitting the data into three parts. The train data is the dataset on which you train your model. Test data is the data on which you, surprise surprise, test your data. But there is a third one, we won't be using it today. It is called evaluate data. It's the last dataset which your model has not ever seen, never been trained on. After you fine-tuned your model, you can typically recombine the train and test data, train your model with it and make a final evaluation with your eval data.

```python
from sklearn.model_selection import train_test_split

x_train, x_eval, y_train, y_eval = train_test_split(x_data,y_true,test_size=0.3, random_state = 101)
```

## Creating input functions

The training instance of the estimator API takes in an input function that you can call with [tf.estimator.inputs.numpy_input_fn](https://www.tensorflow.org/api_docs/python/tf/estimator/inputs/numpy_input_fn).

The x data with a feed dictionary, the y_train data, the batch size, the number of epochs. Shuffle means shuffling through the order of the data that was fed in. Queuing and threading are concepts that are beyond the scope of this workshop.

```python
numpy_input_fn(
    x,
    y=None,
    batch_size=128,
    num_epochs=1,
    shuffle=None,
    queue_capacity=1000,
    num_threads=1
)
```

WORKSHOP MATERIALS SOON TO CONTINUE