# Hunting Through Random Forest for BigFoot: Descending a Ridge to Find the Decomposed Remains

## Raw (Hairy) Data:

The raw data took the form of a JSON file of HMTL code. 

We read it into a dataframe with the following line: 
```python
data = pd.read_json('data/bigfoot_data.json', lines=True)
```

It became clear that the 'html' column was the column of interest, and since it was still a dense thicket of HTML, we cooked up a function using BeautifulSoup to parse the html. This parser took the 'html' column as input and returned a dataframe indexed by report ID, with features and observations corresponding to the key-value pairs of each report. 

## Features: 
* Year, with some data going back as far as the 1800s, up to the present
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

![Top Ten Words for 10-topic NMF Analysis](images/nmf_10topic_chart.png)

Shown another way:
![Top Ten Words for 10-topic NMF Analysis](images/topWordsWordCloud.png)

## Interpreation of NMF Topics:
Topics were fairly distinct and generally described the locational circumstances where the observer made a sighting (Topics 4,5,6,8,9,10). The remainder tended to be some physical description or vestige of the creature itself (Topics 2, 3). Topic 1 was the most generic/vague and difficult to pin down, but the presence of words like "told, know, just, years" may convey a sense of certainty about the story, i.e., "I just know, from my years of experience..."

### Topic Breakdown
* Topic 1 - certainty of encounter
* Topic 2 - auditory descriptions
* Topic 3 - tracks
* Topic 4 - building from which it was encountered
* Topic 5 - road/driving encounters
* Topic 6 - camping encounters
* Topic 7 - description of creature
* Topic 8 - hunting in (random?) forest encounters
* Topic 9 - water/marine encounters
* Topic 10 - trail encounters 



## K Means


After completing the text processing, we applied KMeans clustering along with NMF to the TFIDF vectorized dataset. To see how the two methods compared, we set both cluster/topic hyperparameters to  10. 

Similar to NMF, the key words that made up the 10 groups provided insight into which articles could be used to collect info on Big Foot sightings near roads, wooded areas, homes/buildings, and bodies of water. Reports in group 5 seemed to provide the most details on what Big Foot actually looked like.


Top Words for Each Cluster:
0: lake, heard, area, time, like, trail, just, sound, bigfoot, saw 
1, tree, woods, deer, trail, area, hunting, like, heard, just, time 
2, road, saw, car, just, creature, feet, area, tall, like, right 
3, house, heard, like, woods, saw, just, night, time, went, said 
4, river, heard, saw, like, just, area, time, went, looked, walking
5, saw, looked, like, tall, creature, just, large, feet, seen, hair 
6, heard, sound, like, sounded, night, sounds, loud, noise, hear, area 
7, tracks, area, foot, road, like, long, large, took, saw, day 
8, camp, night, heard, like, went, got, sound, just, time, area 
9, creek, area, heard, like, saw, got, time, just, looked, road


When we mapped these groups back to U.S. states we saw that Group 3-7 were mostly from Washington and Michigan.

Most Common States in Each Cluster:
Group 0: [(' Illinois', 3), (' Michigan', 3), (' Washington', 2), (' Ohio', 1), (' Kansas', 1)]

Group 1: [(' Illinois', 5), (' Washington', 4), (' Michigan', 4), (' Missouri', 4), (' Ohio', 2)]

Group 2: [(' Michigan', 7), (' Illinois', 5), (' Ohio', 4), (' Washington', 4), (' Missouri', 4)]

Group 3: [(' Washington', 13), (' Michigan', 7), (' Georgia', 6), (' Illinois', 5), (' Missouri', 5)]

Group 4: [(' Washington', 17), (' Michigan', 9), (' Illinois', 7), (' Georgia', 7), (' Missouri', 6)]

Group 5: [(' Washington', 19), (' Michigan', 10), (' Georgia', 9), (' Illinois', 8), (' Florida', 7)]

Group 6: [(' Washington', 21), (' Michigan', 10), (' Florida', 9), (' Georgia', 9), (' Illinois', 8)]

Group 7: [(' Washington', 23), (' Michigan', 10), (' Florida', 9), (' Georgia', 9), (' Illinois', 8)]

Group 8: [(' Washington', 27), (' Michigan', 11), (' Georgia', 10), (' Florida', 9), (' Oregon', 9)]

Group 9: [(' Washington', 32), (' Michigan', 13), (' Georgia', 11), (' Illinois', 10), (' Ohio', 9)]





