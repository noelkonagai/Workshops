# Web Scraping Part 1: Importing your JSON data into Pandas

Now that you have learned how to use Yelp API as well as how to use Pandas, let's combine the two together. You have a JSON file, and let's use this JSON file in Pandas.

## 1. Set up a new notebook

Just write our canonical import functions.

```python
import pandas as pd
import json
```
## 2. Open the JSON file

This is a little more tricky. Depending on how you have saved the file, you might get some errors. Remember, when I opened my ```data.json``` file, I used ```'w'``` as a method to open it.

Write the following.

```python
with open('data.json') as openfile:
	data = json.load(openfile)
```

Now your json file is stored in the variable ```data```. Try experimenting a little bit, what does the print out look like?

```python
print(data)
```

Quite messy, right?

If you just want to study your data in its raw format, you can import a library called ```pprint```. The first p stands for 'pretty' because it prettifies the print out of this JSON file.

```python
import pprint

pprint.pprint(data)
```
Now the output looks slightly better. But still not manageable for Pandas. You could try ```pd.read_json(data)``` and you will get an error.

We need to get it down to a format that is understandable for Pandas. The 'categories' and 'coordinates' cause the error for us since they are nested dictionaries that is a dimension higher than what Pandas can handle. Pause here and try to think what could be the solution for this.

First, we could get rid of all the categories and coordinates, but then we lose data. Or we could compromise and treat them as a string instead of nested dictionaries. Try the following to create a Pandas DataFrame out of our JSON input.

```python
df = pd.DataFrame(data['businesses'])
```

Now, this looks much better! Just to demonstrate how easy it is to create DataFrames try to create a DataFrame from the 0th entry in categories.

```python
pd.DataFrame(df['categories'][0])
```

Congratulations! You finished this workshop.


