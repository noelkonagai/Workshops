# TensorFlow: Wide & Deep Learning API

In this workshop I will introduce you to TensorFlow's Wide and Deep Learning model. We will be using TensorFlow's tutorial files. This workshop will comment on each section of the TensorFlow tutorial file. You can expect to have an overview and explanation on how to call on certain APIs to process your data for a future project.

![Image of a Wide and Deep Network]
(https://www.tensorflow.org/images/wide_n_deep.svg)

Above is an image of a Wide and Deep Network. This type of combined structure combines linear regressions' so called "memorization" feature and deep neural networks' "generalization" feature. The quote below from this [research paper](https://arxiv.org/abs/1606.07792) from explains what these two concepts mean.

>"Memorization can be loosely defined as learning the frequent co-occurrence of items or features and exploiting the correlation available in the historical data. Generalization, on the other hand, is based on transitivity of correlation and explores new feature combinations that have never or rarely occurred in the past."

To start, download and run [data_download.py](https://minhaskamal.github.io/DownGit/#/home?url=https://github.com/noelkonagai/Workshops/blob/master/tensorflow_pt2_widedeep/data_download.py) that will download ```adult.data``` and ```adult.test``` files. Also, download [wide_deep.py](https://github.com/noelkonagai/Workshops/blob/master/tensorflow_pt2_widedeep/wide_deep.py), which you will run at the end of the workshop.

## 1. Choice of Model: DNNLinearCombinedClassifier

The model that we are going to use is called ```DNNLinearCombinedClassifier```. This not only combines the "wide" and "deep" models but also trains them together. There are number of other models, which all work slightly differently. Depending on your project, you may use any of the following non-exhaustive list.

```LinearClassifier```
```LinearRegressor```
```LogisticRegressor```
```KMeansClustering```
```DNNClassifier```
```DNNRegressor```

You may read up on these at [tf.contrib.learn](https://www.tensorflow.org/api_docs/python/tf/contrib/learn).

We will choose ```DNNLinearCombinedClassifier``` for our dataset of [Census Income Dataset](https://archive.ics.uci.edu/ml/datasets/Census+Income) to predict whether the income with given age, education and more features will exceed 50,000 USD per year. 

## 2. Setting up linear feature columns

There are a number of types of feature column that you can set up in your model. We will be focusing on five different types.

### tf.feature_column.numeric_column

This API is for columns that contain continuous values, such as income, price of a given product, rent price, and so forth. You may also treat discrete valued columns as continuous values, as the TF tutorial does for "age". 

```python
age = tf.feature_column.numeric_column('age')
education_num = tf.feature_column.numeric_column('education_num')
capital_gain = tf.feature_column.numeric_column('capital_gain')
capital_loss = tf.feature_column.numeric_column('capital_loss')
hours_per_week = tf.feature_column.numeric_column('hours_per_week')
```

### tf.feature_column.categorical_column_with_vocabulary_list

This API is used for categorical or discrete values and we pass the vocabulary list as an argument. 

```python
education = tf.feature_column.categorical_column_with_vocabulary_list(
    'education', [
        'Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
        'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
        '5th-6th', '10th', '1st-4th', 'Preschool', '12th'])
```

### Q: Can you find a way to input the vocabulary list into these columns?

```
marital_status = tf.feature_column.categorical_column_with_vocabulary_list(
    'marital_status', [<vocab list>])

relationship = tf.feature_column.categorical_column_with_vocabulary_list(
    'relationship', [<vocab list>])

workclass = tf.feature_column.categorical_column_with_vocabulary_list(
    'workclass', [<vocab list>])
```

### tf.feature_column.categorical_column_with_vocabulary_file

If your list is too long, you can also use a vocabulary file in the following way.

```python
tf.feature_column.categorical_column_with_vocabulary_file(
        key=feature_name_from_input_fn,
        vocabulary_file="vocab.txt",
        vocabulary_size=3)
```
```vocab.txt``` should contain one line for each vocabulary element.

### tf.feature_column.categorical_column_with_hash_bucket

When you have too many categories to list out you can use a hash bucket. It assigns an ID to each of the distinct discrete valued element in the column. It requires a ```hash_bucket_size``` argument, which is the number of the most amounts of hashes you wish to have.

```python
occupation = tf.feature_column.categorical_column_with_hash_bucket(
    'occupation', hash_bucket_size=1000)
```

### tf.feature_column.bucketized_column

When you wish to group together certain values then you can use this API. For instance, if you are interested in age groups rather than specific ages then you can bucketize your column. The first argument you pass on is the source column, second is boundaries. Note: you already have to have a linear column to make a bucketized column out of it.

```python
age_buckets = tf.feature_column.bucketized_column(
    age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
```

Buckets include the left boundary, exclude the right. Namely, boundaries=[18, 25] generates buckets (-inf, 18), [18, 25), and [25, +inf).

## 3. Base Columns and Crossed Columns

### base_columns

You will need to create a list out of your base columns.

```python
base_columns = [
    education, marital_status, relationship, workclass, occupation,
    age_buckets,
]
```

### crossed_columns

The crossed columns can memorize the sparse interactions between the features you input. So for instance, if you think that income and university degree have something in common you may cross these columns. In the case of our data, it is the following. The first argument is the column ```key``` in a list, next is the ```hash_bucket_size```.

```python
crossed_columns = [
    tf.feature_column.crossed_column(
        ['education', 'occupation'], hash_bucket_size=1000),
    tf.feature_column.crossed_column(
        [age_buckets, 'education', 'occupation'], hash_bucket_size=1000),
]
```

## 4. Embedded Columns for Deep Learning

Here we will pass in the continuous columns together with one-hot, multi-hot encoded, or embedded categorical columns.

### tf.feature_column.indicator_column

Say, you have n categories. A function converts each element into a vector of n entries by placing a 1 in the index of the categorical value, and 0 everywhere else. Say you have three categories for height: tall, medium, short. You will have [1,0,0], [0,1,0], [0,0,1] representations for each of these categories.

### tf.feature_column.embedding_column

A function uses the numerical categorical values as indices to a lookup table. Each slot in that lookup table contains a n-element vector. You can define what n is if you pass it as a dimension argument.

This is how it would look like in our case:

```python
deep_columns = [
    age,
    education_num,
    capital_gain,
    capital_loss,
    hours_per_week,
    tf.feature_column.indicator_column(workclass),
    tf.feature_column.indicator_column(education),
    tf.feature_column.indicator_column(marital_status),
    tf.feature_column.indicator_column(relationship),
    tf.feature_column.embedding_column(occupation, dimension=8),
]
```

## 4. Creating the model

We are using ```tf.estimator.DNNLinearCombinedClassifier``` to create our model. You can set a ```model_dir``` where the model will be saved. ```dnn_hidden_units``` refers to the number of hidden neurons in each layer. [100,50] is two hidden layers with the first having 100 neurons, the second one having 50 neurons.

```python
model = tf.estimator.DNNLinearCombinedClassifier(
    model_dir='/census_model',
    linear_feature_columns= base_columns + crossed_columns,
    dnn_feature_columns= deep_columns,
    dnn_hidden_units=[100, 50])
```

## 5. Run the sample files

Run ```wide_deep.py```.

This is the end of the workshop.

