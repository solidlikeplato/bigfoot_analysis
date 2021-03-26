

# Raw (Hairy) Data:

The raw data took the form of a JSON file of HMTL code. 

We read it into a dataframe with the following line: 
```python
data = pd.read_json('data/bigfoot_data.json', lines=True)
```

It became clear that the 'html' column was the column of interest, and since it was still a dense thicket of HTML, we cooked up a function using BeautifulSoup to parse the html. This parser took the 'html' column as input and returned a dataframe indexed by report ID, with features and observations corresponding to the key-value pairs of each report. 

## Features: 
* Year
* Season
* Month
* State
* Location Details
* Nearest Town
* Nearest Road
* Observed
* Also Noticed
* Other Witnesses
* Other Stories
* Time and Conditions
* Environment


# Details of Text Processing Pipeline:

The text of interest was found in the "Observed" column of the dataframe, which contains the witnesse's description of the encounter. It was processed using tfidf after removing English stopwords and converting the data to unicode. 

# ML Algorithms

## NMF
We used an implementation of NMF to produce a topic analysis on the Observations. Analyzing for 10 componenents with max_iter of 500, the nmf model was able to identify distinct topics that generally described the circumstances in which the observer saw the creature:





## K Means



