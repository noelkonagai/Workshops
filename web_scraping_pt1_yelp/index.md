# Web Scraping Part 1: Using Yelp API to get JSON data

Yelp is an application, which crowdsources reviews of venues such as restaurants, cafes, and more. We will be using Yelp API (Application Programming Interface) to collect data with sample search queries.

## 1. Sign up for Yelp

In order to use the Yelp API, you need to sign up via this [link](https://www.yelp.com/login?return_url=%2Fdevelopers%2Fv3%2Fmanage_app). Select Hong Kong as your location as China is not available.

Create an application, set a name such as ```test_application``` and choose any Industry. Then confirm the sign up via your email that you have used as a contact email. Click create app.

Now you've received an API key. An API key is a long string that is used for identification purposes. You will be using this key to make query requests. Save the Client ID, API key, and Client Secret in a text file on your computer. You're given a daily limit of making 25000 API calls. While it seems a lot, if you're an app developer and want to make your app publicly available, quickly these API calls will turn out to be not enough. That is why you should keep your API key and other identification strings secret.

## 2. Make a query request in Python

Navigate yourself to this [Github Repository](https://github.com/noelkonagai/yelp-fusion/tree/master/fusion/python) and download the contents of the repository. If you're unsure whether you have the right dependencies, follow the Readme.MD instructions, otherwise you can open ```sample.py``` and tweak the code a little bit.

In line 48 add your own API key, in between parentheses.

Now navigate in the command line to the folder that contains ```sample.py``` and run ```python sample.py```.

You will receive back a JSON file printed out in the shell. The sample query uses the keyword "dinner" to search for any venues in 'San Francisco, CA' and shows the top three results. Lines 58 to 60 show these.

```python
DEFAULT_TERM = 'dinner'
DEFAULT_LOCATION = 'San Francisco, CA'
SEARCH_LIMIT = 3
```

A handy feature that was included when Yelp developers wrote ```sample.py``` was a functionality of using arguments after two dashed lines. Run in the command line the following.

```python
python sample.py --term="bars" --location="San Francisco, CA"
```

Now it shows you bars in San Francisco. The same command can be written shorter.

```python
python sample.py -q="bars" -l="San Francisco, CA"
```

## 3. Saving the JSON output

You can either copy-paste or tweak the code so that it saves your JSON return file. If you're adventurous, follow the instructions below in learning how to save the JSON file.

Currently, with our settings, it only saves three responses. With the Yelp API, you can raise your search limit up to 50. In order to get more than 50, you need to set an offset variable. Let's do this.

Change lines 58 to 60.

```python
DEFAULT_TERM = 'dinner'
DEFAULT_LOCATION = 'San Francisco, CA'
SEARCH_LIMIT = 50
OFFSET = 0
```

You have raised the ```SEARCH_LIMIT``` to its maximum 50. This way you will get the listings from 0 through 49. To get the listings from 50 through 99, you can change the ```OFFSET``` to 50.

Now, to line 133 after ```response = search(API_KEY, term, location)``` add the following.

```python
with open('data.json', 'w') as openfile:
	json.dump(response, openfile)
```

With the ```json.dump()``` function you are saving the JSON response into a file that you specify with the ```with open('filepath')``` clause.

Now run again the following:

```python
python sample.py -q="bars" -l="San Francisco, CA"
```

And look now at your folder where ```sample.py``` is. You see now a ```data.json``` file that contains the JSON response file! It looks quite messy, and it should look like this but our pprint library prettified it for us. Now, this is not the only thing that Yelp API can do. For more details and to appease your curiosity, read the [ Official Documentation](https://www.yelp.com/developers/documentation/v3/get_started).