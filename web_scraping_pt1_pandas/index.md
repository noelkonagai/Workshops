# Web Scraping Part 1: Introduction to Pandas

Pandas are cute animals that you can find in the zoo of Chengdu. True, but Pandas is also your best friend when you want to deal with data in Python. You can import all sorts of files, and for the purposes of this workshop we will import a CSV Housing data file that contains information on household value, median household age, household's proximity to the ocean, etc. Once we have imported this data, it will become a Dataframe object. Each column in the Dataframe is called a Series. Now why is Pandas our best friend when dealing with data? It cuts down our runtime for lot of the functions, such as searching functions, that we would need to write manually.

## 1. Setting up Pandas

With the following options you can set the amount of rows and columns that you want the Jupyter notebook to display.

```python
import pandas as pd
pd.set_option('display.max_columns', 30) 
pd.options.display.max_rows = 20
```

## 2. Importing raw data

Download the housing data collected from this [source](https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv).

```python
data = pd.read_csv("housing.csv")
```

Now you have a Pandas dataframe called data. We will explore what you can do with the *dataframe*.

## 3. Basic operations with dataframes

To diplay just the first ficve rows of the dataframe:

```python
data[:5]
```

To list out the column headers of the dataframe:

```python
data.columns.values.tolist()
```

And to get a basic statistical description of your dataframe, such as the number count for each column, mean, standard deviation, percentiles, min and max:

```python
data.describe()
```

To select just one column and make it a Pandas *series*:

```python
population = data['population']
```

To select multiple columns:

```python
data[['population', 'households']]
```

## 3. Let's answer basic questions using Pandas

### What is the most common number of rooms in a house?

You can use ```.value_counts()``` after a column to check the frequency of each descrete entity. Typically this is better for a finite number of values, such as "height of the building", which could have high, medium, and low as entries.

```python
data['total_rooms'].value_counts()
```

The answer is 1527 rooms.

### How many listings are close to the ocean?

```python
data['ocean_proximity'].value_counts()
```

The answer is 2658 if you do not count the ones that are on an island.

## 4. Why is Pandas so good though?

It's pretty much hidden from your sight. Maybe you have used Excel to do similar tasks before. Or maybe you can write search functions with for and conditional loops in Python. So let's see if Pandas performs better than for loops!

### Using for loops vs. Pandas

Here is a sample code with for loops to find the listings where "ocean_proximity" is "ISLAND". Use ```%%time``` to display the time taken by your computer to compute the Jupyter Notebook cell output. 

```python
%%time

island_indices = []

for i in range(len(data['ocean_proximity'])):
    if data['ocean_proximity'][i] == 'ISLAND':
        island_indices.append(i)
        
for index in island_indices:
    print(index, data.as_matrix()[index])
```

For me the output is about 300 ms.

Let's do the same with Pandas.

```python
%%time

island_lots = data[data['ocean_proximity'] == "ISLAND"]

print(island_lots)
```

The time is approximately 13 ms, only 4.3% of the time that it took to find with for loops the listings that are on Islands.

### Writing a search function with lower and upper bounds.

This is a sample code using for loops:

```python
def forloop_search_function(lower, upper, column, data):
    indices = []
    
    for i in range(len(data)):
        if data[str(column)][i] > lower and data[str(column)][i] < upper:
            indices.append(i)
            
    return indices
```

Then run it in another cell with:

```python
%%time

forloop_search_function(26, 30, "housing_median_age", data)
```

The time it took for me is about 450 ms.

Let's write the same function with Pandas.

```python
def pandas_search_function(lower, upper, column, data):
    first_cond = data[data[str(column)] > lower]
    combined = first_cond[first_cond[str(column)] < upper]
    return combined
```

And run it with:

```python
%%time

narrowed_df = search_function(26, 30, "housing_median_age", data)
```

It took about 4 ms. With for loops this takes more than 100 times longer! Crazy!

To summarize, Pandas makes your life easier when you deal with bigger data. With this small data, you will not notice much difference in the runtime, but once your data becomes larger (and trust me it will), Pandas will noticeably perform better.

## 5. Sorting dataframes with Pandas

Again, you could sort with your manually written functions, but in Pandas you can call the handy  ```.sort_values()``` on a Dataframe. You will probably use these three arguments below, for the rest check out the official [API guide](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sort_values.html).

by -- string value of the column name by which you want to sort
axis -- direction of sorting, 0 for sortying by the ‘index’, 1 for sorting by ‘columns’, defaults to 0
ascending -- boolean value True/False

Let's sort our dataframe by the median household age.

```python
data.sort_values('housing_median_age')[:5]
```

If we want the oldest houses first:

```python
data.sort_values('housing_median_age', ascending = False)[:5]
```
