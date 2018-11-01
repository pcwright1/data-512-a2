# data-512-a2
Data 512 A2 Assignment -- Bias on Wikipedia

This ipython notebook is created for DATA512 at UW for this assignment: https://wiki.communitydata.cc/HCDS_(Fall_2017)/Assignments#A2:_Bias_in_data

Our goal is to analyze the content of wikipedia to understand the biases of the site by looking at the content coverage for political members of countries. We look at how many pages there are (as a percent of the country's population) and how many of the pages are high quality (using scores from the ORES system, more info below).

In the end, we show the top/bottom 10 countries for these 2 categories.

## Related Data Files

raw data files:
- page_data.csv : raw wikipedia data
- WPDS_2018_data.csv : raw country population data

Output files:
- ores_data.csv : articles scores from the ORES system
- combined_data.csv : combined data (country population, ores data and wikipedia data)


First, import necessary packages


```python
import requests
import json
import pandas as pd
import numpy as np


```

Import the data, and print out the first few rows to see examples.

Data comes from a few different sources. Wikipedia data is available via figshare (https://figshare.com/articles/Untitled_Item/5513449 , under country/data/) with license CC-BY-SA 4.0. This contains "most English-language Wikipedia articles within the category 'Category:Politicians by nationality' and subcategories". This data contains 3 columns, which are called out in the above link as follows:

1. "country", containing the sanitised country name, extracted from the category name;
2. "page", containing the unsanitised page title.
3. "last_edit", containing the edit ID of the last edit to the page.

Population data is available via https://www.dropbox.com/s/5u7sy1xt7g0oi2c/WPDS_2018_data.csv?dl=0.
This file contains the population in millions from mid-2018 along with the country name.

A copy of the datasets, downloaded in oct, 2018, are available in this repo.


```python
wiki_data = pd.read_csv('page_data.csv')
country_data = pd.read_csv('WPDS_2018_data.csv',thousands=',')
country_data.rename(columns={"Population mid-2018 (millions)": "population"},inplace=True)

wiki_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>page</th>
      <th>country</th>
      <th>rev_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Template:ZambiaProvincialMinisters</td>
      <td>Zambia</td>
      <td>235107991</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bir I of Kanem</td>
      <td>Chad</td>
      <td>355319463</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Template:Zimbabwe-politician-stub</td>
      <td>Zimbabwe</td>
      <td>391862046</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Template:Uganda-politician-stub</td>
      <td>Uganda</td>
      <td>391862070</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Template:Namibia-politician-stub</td>
      <td>Namibia</td>
      <td>391862409</td>
    </tr>
  </tbody>
</table>
</div>




```python
country_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Geography</th>
      <th>population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AFRICA</td>
      <td>1284.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Algeria</td>
      <td>42.7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Egypt</td>
      <td>97.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Libya</td>
      <td>6.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Morocco</td>
      <td>35.2</td>
    </tr>
  </tbody>
</table>
</div>



Here we create a helper function for getting ores scores

This function takes revision ids (and the headers needed to make the call) and scores the function using the ORES system. The score and the revision id are appended to the ores_data list.

ORES (Objective Revision Evaluation Service) is a machine learning service that ranks the quality of a given article. The ranks go from best to worst as FA, GA, B, C, Start and Stub. For the purposes of this analysis, we use only the predicted category (rather than the probabilities, which are also available).
link with more info: https://www.mediawiki.org/wiki/ORES




```python

def get_ores_data(revision_ids, headers):
    temp_data = []
    # Define the endpoint
    endpoint = 'https://ores.wikimedia.org/v3/scores/{project}/?models={model}&revids={revids}'
    
    params = {'project' : 'enwiki',
              'model'   : 'wp10',
              'revids'  : '|'.join(str(x) for x in revision_ids)
              }
    api_call = requests.get(endpoint.format(**params))
    response = pd.read_json(json.dumps(api_call.json(), indent=4, sort_keys=True))
    for id in response['enwiki']['scores']:
        try:
            ores_data.append([id, response['enwiki']['scores'][id]['wp10']['score']['prediction']])
        except:
            pass
    
    #print(json.dumps(response, indent=4, sort_keys=True))
    #return temp_data #response
```

Here we define the header needed to call the above function and iterate over all of the revions, calling the function in batches (of about 100, or 472 batches for slightly less than 47k revisions).


```python
%%time
# So if we grab some example revision IDs and turn them into a list and then call get_ores_data...
ores_data = [] #pd.DataFrame(columns =['revid','category'])
#ores_data.append([['a','b']])
#print(ores_data)
headers = {'User-Agent' : 'https://github.com/your_github_username', 'From' : 'your_uw_email@uw.edu'}
for i in np.array_split(np.asarray(wiki_data['rev_id']),472): #, 472): #split into buckets of approximately 100
    get_ores_data(i, headers)#,columns =['revid','category']
    #temp_data = pd.DataFrame(get_ores_data(i, headers),columns =['revid','category'])
    
    #print("here")
    #print(ores_data)
    #print(temp_data)
    #ores_data.append(temp_data)
```

    CPU times: user 14.1 s, sys: 664 ms, total: 14.7 s
    Wall time: 2min 24s


Here we convert the ores_data into a pandas dataframe and save to a csv for reference.


```python
ores_data = pd.DataFrame(ores_data,columns =['revision_id','article_quality'])#.set_index('revision_id')
ores_data.to_csv('ores_data.csv')
```

We convert revision_id to a int so we can join it to the wikipedia data.


```python
#check out ores
ores_data['revision_id'] = ores_data['revision_id'].astype(int)
#ores_data.set_index('revid')
#ores_data.reset_index(inplace=True)
ores_data.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>revision_id</th>
      <th>article_quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>355319463</td>
      <td>Stub</td>
    </tr>
    <tr>
      <th>1</th>
      <td>391862046</td>
      <td>Stub</td>
    </tr>
    <tr>
      <th>2</th>
      <td>391862070</td>
      <td>Stub</td>
    </tr>
    <tr>
      <th>3</th>
      <td>391862409</td>
      <td>Stub</td>
    </tr>
    <tr>
      <th>4</th>
      <td>391862819</td>
      <td>Stub</td>
    </tr>
  </tbody>
</table>
</div>



Here we merge the wikipedia data to the ores data on the revision id. We also merge onto the country data on the country/geography columns. There are 44,973 rows left after we inner join.


```python
# Merge data
combined_data = wiki_data.merge(country_data,
                                how = 'inner',
                                left_on ='country',
                                right_on = 'Geography').merge(ores_data,
                                                              how = 'inner',
                                                              left_on = 'rev_id',
                                                              right_on = 'revision_id'
                                                             )
print(combined_data.shape)

```

    (44973, 7)


Here is a preview of the US data:


```python
combined_data[combined_data['country']=='United States'].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>page</th>
      <th>country</th>
      <th>rev_id</th>
      <th>Geography</th>
      <th>population</th>
      <th>revision_id</th>
      <th>article_quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>26912</th>
      <td>Butler-Belmont family</td>
      <td>United States</td>
      <td>470173494</td>
      <td>United States</td>
      <td>328.0</td>
      <td>470173494</td>
      <td>Start</td>
    </tr>
    <tr>
      <th>26913</th>
      <td>Heard-Hawes family</td>
      <td>United States</td>
      <td>502721672</td>
      <td>United States</td>
      <td>328.0</td>
      <td>502721672</td>
      <td>C</td>
    </tr>
    <tr>
      <th>26914</th>
      <td>Russell family (American political family)</td>
      <td>United States</td>
      <td>550953646</td>
      <td>United States</td>
      <td>328.0</td>
      <td>550953646</td>
      <td>Stub</td>
    </tr>
    <tr>
      <th>26915</th>
      <td>Read family of Delaware</td>
      <td>United States</td>
      <td>651856758</td>
      <td>United States</td>
      <td>328.0</td>
      <td>651856758</td>
      <td>Start</td>
    </tr>
    <tr>
      <th>26916</th>
      <td>Template:US-politician-stub</td>
      <td>United States</td>
      <td>666834672</td>
      <td>United States</td>
      <td>328.0</td>
      <td>666834672</td>
      <td>Stub</td>
    </tr>
  </tbody>
</table>
</div>



We filter the new dataset to remove duplicate columns and save this to a csv.


```python
combined_data = combined_data[['country','page','revision_id','article_quality','population']]
combined_data.to_csv('combined_data.csv')
```

## Analysis

Here we start analysing the data. First, we create a pivot table with population by country.


```python
# Analysis
articles_and_population = combined_data.pivot_table(values = ['population'],
                          index = ['country'],
                          dropna = False,
                          #columns = ['article_quality'],
                          aggfunc = {'population': min,'country':'count'}
                         ).rename(columns={"country": "num_articles"}).reset_index()
articles_and_population.shape
```




    (180, 3)



Next, we create a pivot table with number of high quality articles by country.


```python
high_qual_articles = combined_data[combined_data['article_quality'].isin(['FA','GA'])].pivot_table(values = ['population'],
                          index = ['country'],
                          dropna = False,
                          #columns = ['article_quality'],
                          aggfunc = {'country':'count'}
                         ).rename(columns={"country": "num_high_quality_articles"}).reset_index()
high_qual_articles.shape
```




    (143, 2)



We join the datasets and fill NAs with zeros. We change num_articles to be an int and population to be a float.

We then calculate the articles_per_population (which is per million people) and the high quality article percentage for each country.

Finally, we set the index as the country (as these are unique) and display the results.


```python
dataset = articles_and_population.merge(high_qual_articles, how='left').fillna(0)
dataset['num_articles'] = dataset['num_articles'].astype(int)
dataset['population'] = dataset['population'].astype(float)



#dataset.dropna(inplace=True)
dataset['articles_per_population'] = dataset['num_articles'] / dataset['population']
dataset['high_qual_article_perc'] = dataset['num_high_quality_articles'] / dataset['num_articles']
dataset.set_index('country',inplace=True)
dataset
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>num_articles</th>
      <th>population</th>
      <th>num_high_quality_articles</th>
      <th>articles_per_population</th>
      <th>high_qual_article_perc</th>
    </tr>
    <tr>
      <th>country</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Afghanistan</th>
      <td>326</td>
      <td>36.50</td>
      <td>10.0</td>
      <td>8.931507</td>
      <td>0.030675</td>
    </tr>
    <tr>
      <th>Albania</th>
      <td>460</td>
      <td>2.90</td>
      <td>4.0</td>
      <td>158.620690</td>
      <td>0.008696</td>
    </tr>
    <tr>
      <th>Algeria</th>
      <td>119</td>
      <td>42.70</td>
      <td>2.0</td>
      <td>2.786885</td>
      <td>0.016807</td>
    </tr>
    <tr>
      <th>Andorra</th>
      <td>34</td>
      <td>0.08</td>
      <td>0.0</td>
      <td>425.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Angola</th>
      <td>110</td>
      <td>30.40</td>
      <td>0.0</td>
      <td>3.618421</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Antigua and Barbuda</th>
      <td>25</td>
      <td>0.10</td>
      <td>0.0</td>
      <td>250.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Argentina</th>
      <td>496</td>
      <td>44.50</td>
      <td>15.0</td>
      <td>11.146067</td>
      <td>0.030242</td>
    </tr>
    <tr>
      <th>Armenia</th>
      <td>198</td>
      <td>3.00</td>
      <td>5.0</td>
      <td>66.000000</td>
      <td>0.025253</td>
    </tr>
    <tr>
      <th>Australia</th>
      <td>1566</td>
      <td>24.10</td>
      <td>42.0</td>
      <td>64.979253</td>
      <td>0.026820</td>
    </tr>
    <tr>
      <th>Austria</th>
      <td>340</td>
      <td>8.80</td>
      <td>3.0</td>
      <td>38.636364</td>
      <td>0.008824</td>
    </tr>
    <tr>
      <th>Azerbaijan</th>
      <td>182</td>
      <td>9.90</td>
      <td>2.0</td>
      <td>18.383838</td>
      <td>0.010989</td>
    </tr>
    <tr>
      <th>Bahamas</th>
      <td>20</td>
      <td>0.40</td>
      <td>0.0</td>
      <td>50.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Bahrain</th>
      <td>42</td>
      <td>1.50</td>
      <td>1.0</td>
      <td>28.000000</td>
      <td>0.023810</td>
    </tr>
    <tr>
      <th>Bangladesh</th>
      <td>323</td>
      <td>166.40</td>
      <td>3.0</td>
      <td>1.941106</td>
      <td>0.009288</td>
    </tr>
    <tr>
      <th>Barbados</th>
      <td>14</td>
      <td>0.30</td>
      <td>0.0</td>
      <td>46.666667</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Belarus</th>
      <td>72</td>
      <td>9.50</td>
      <td>2.0</td>
      <td>7.578947</td>
      <td>0.027778</td>
    </tr>
    <tr>
      <th>Belgium</th>
      <td>523</td>
      <td>11.40</td>
      <td>0.0</td>
      <td>45.877193</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Belize</th>
      <td>16</td>
      <td>0.40</td>
      <td>0.0</td>
      <td>40.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Benin</th>
      <td>94</td>
      <td>11.50</td>
      <td>7.0</td>
      <td>8.173913</td>
      <td>0.074468</td>
    </tr>
    <tr>
      <th>Bhutan</th>
      <td>33</td>
      <td>0.80</td>
      <td>3.0</td>
      <td>41.250000</td>
      <td>0.090909</td>
    </tr>
    <tr>
      <th>Bolivia</th>
      <td>187</td>
      <td>11.30</td>
      <td>1.0</td>
      <td>16.548673</td>
      <td>0.005348</td>
    </tr>
    <tr>
      <th>Bosnia-Herzegovina</th>
      <td>177</td>
      <td>3.50</td>
      <td>7.0</td>
      <td>50.571429</td>
      <td>0.039548</td>
    </tr>
    <tr>
      <th>Botswana</th>
      <td>68</td>
      <td>2.20</td>
      <td>2.0</td>
      <td>30.909091</td>
      <td>0.029412</td>
    </tr>
    <tr>
      <th>Brazil</th>
      <td>551</td>
      <td>209.40</td>
      <td>3.0</td>
      <td>2.631328</td>
      <td>0.005445</td>
    </tr>
    <tr>
      <th>Bulgaria</th>
      <td>226</td>
      <td>7.00</td>
      <td>3.0</td>
      <td>32.285714</td>
      <td>0.013274</td>
    </tr>
    <tr>
      <th>Burkina Faso</th>
      <td>97</td>
      <td>20.30</td>
      <td>3.0</td>
      <td>4.778325</td>
      <td>0.030928</td>
    </tr>
    <tr>
      <th>Burundi</th>
      <td>76</td>
      <td>11.80</td>
      <td>1.0</td>
      <td>6.440678</td>
      <td>0.013158</td>
    </tr>
    <tr>
      <th>Cambodia</th>
      <td>217</td>
      <td>16.00</td>
      <td>3.0</td>
      <td>13.562500</td>
      <td>0.013825</td>
    </tr>
    <tr>
      <th>Cameroon</th>
      <td>105</td>
      <td>25.60</td>
      <td>0.0</td>
      <td>4.101562</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Canada</th>
      <td>850</td>
      <td>37.20</td>
      <td>25.0</td>
      <td>22.849462</td>
      <td>0.029412</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>Sri Lanka</th>
      <td>465</td>
      <td>21.70</td>
      <td>10.0</td>
      <td>21.428571</td>
      <td>0.021505</td>
    </tr>
    <tr>
      <th>Sudan</th>
      <td>98</td>
      <td>41.70</td>
      <td>1.0</td>
      <td>2.350120</td>
      <td>0.010204</td>
    </tr>
    <tr>
      <th>Suriname</th>
      <td>40</td>
      <td>0.60</td>
      <td>1.0</td>
      <td>66.666667</td>
      <td>0.025000</td>
    </tr>
    <tr>
      <th>Sweden</th>
      <td>379</td>
      <td>10.20</td>
      <td>5.0</td>
      <td>37.156863</td>
      <td>0.013193</td>
    </tr>
    <tr>
      <th>Switzerland</th>
      <td>407</td>
      <td>8.50</td>
      <td>0.0</td>
      <td>47.882353</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Syria</th>
      <td>132</td>
      <td>18.30</td>
      <td>8.0</td>
      <td>7.213115</td>
      <td>0.060606</td>
    </tr>
    <tr>
      <th>Taiwan</th>
      <td>503</td>
      <td>23.60</td>
      <td>9.0</td>
      <td>21.313559</td>
      <td>0.017893</td>
    </tr>
    <tr>
      <th>Tajikistan</th>
      <td>39</td>
      <td>9.10</td>
      <td>1.0</td>
      <td>4.285714</td>
      <td>0.025641</td>
    </tr>
    <tr>
      <th>Tanzania</th>
      <td>408</td>
      <td>59.10</td>
      <td>1.0</td>
      <td>6.903553</td>
      <td>0.002451</td>
    </tr>
    <tr>
      <th>Thailand</th>
      <td>112</td>
      <td>66.20</td>
      <td>3.0</td>
      <td>1.691843</td>
      <td>0.026786</td>
    </tr>
    <tr>
      <th>Togo</th>
      <td>65</td>
      <td>8.00</td>
      <td>2.0</td>
      <td>8.125000</td>
      <td>0.030769</td>
    </tr>
    <tr>
      <th>Tonga</th>
      <td>63</td>
      <td>0.10</td>
      <td>1.0</td>
      <td>630.000000</td>
      <td>0.015873</td>
    </tr>
    <tr>
      <th>Trinidad and Tobago</th>
      <td>28</td>
      <td>1.40</td>
      <td>1.0</td>
      <td>20.000000</td>
      <td>0.035714</td>
    </tr>
    <tr>
      <th>Tunisia</th>
      <td>140</td>
      <td>11.60</td>
      <td>0.0</td>
      <td>12.068966</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Turkey</th>
      <td>353</td>
      <td>81.30</td>
      <td>4.0</td>
      <td>4.341943</td>
      <td>0.011331</td>
    </tr>
    <tr>
      <th>Turkmenistan</th>
      <td>33</td>
      <td>5.90</td>
      <td>0.0</td>
      <td>5.593220</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Tuvalu</th>
      <td>55</td>
      <td>0.01</td>
      <td>5.0</td>
      <td>5500.000000</td>
      <td>0.090909</td>
    </tr>
    <tr>
      <th>Uganda</th>
      <td>188</td>
      <td>44.10</td>
      <td>0.0</td>
      <td>4.263039</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Ukraine</th>
      <td>304</td>
      <td>42.30</td>
      <td>15.0</td>
      <td>7.186761</td>
      <td>0.049342</td>
    </tr>
    <tr>
      <th>United Arab Emirates</th>
      <td>59</td>
      <td>9.50</td>
      <td>2.0</td>
      <td>6.210526</td>
      <td>0.033898</td>
    </tr>
    <tr>
      <th>United Kingdom</th>
      <td>865</td>
      <td>66.40</td>
      <td>57.0</td>
      <td>13.027108</td>
      <td>0.065896</td>
    </tr>
    <tr>
      <th>United States</th>
      <td>1092</td>
      <td>328.00</td>
      <td>82.0</td>
      <td>3.329268</td>
      <td>0.075092</td>
    </tr>
    <tr>
      <th>Uruguay</th>
      <td>290</td>
      <td>3.50</td>
      <td>2.0</td>
      <td>82.857143</td>
      <td>0.006897</td>
    </tr>
    <tr>
      <th>Uzbekistan</th>
      <td>29</td>
      <td>32.90</td>
      <td>1.0</td>
      <td>0.881459</td>
      <td>0.034483</td>
    </tr>
    <tr>
      <th>Vanuatu</th>
      <td>60</td>
      <td>0.30</td>
      <td>3.0</td>
      <td>200.000000</td>
      <td>0.050000</td>
    </tr>
    <tr>
      <th>Venezuela</th>
      <td>135</td>
      <td>31.80</td>
      <td>3.0</td>
      <td>4.245283</td>
      <td>0.022222</td>
    </tr>
    <tr>
      <th>Vietnam</th>
      <td>191</td>
      <td>94.70</td>
      <td>13.0</td>
      <td>2.016895</td>
      <td>0.068063</td>
    </tr>
    <tr>
      <th>Yemen</th>
      <td>122</td>
      <td>28.90</td>
      <td>2.0</td>
      <td>4.221453</td>
      <td>0.016393</td>
    </tr>
    <tr>
      <th>Zambia</th>
      <td>25</td>
      <td>17.70</td>
      <td>0.0</td>
      <td>1.412429</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Zimbabwe</th>
      <td>167</td>
      <td>14.00</td>
      <td>2.0</td>
      <td>11.928571</td>
      <td>0.011976</td>
    </tr>
  </tbody>
</table>
<p>180 rows × 5 columns</p>
</div>



Finally, display the top and bottome countries by articles per million people. Tuvalu has the highest value, but does have an extremely small population. Of the represented countries, India has the smallest article per million people.


```python
dataset.sort_values(by = 'articles_per_population',ascending = False)[0:10]

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>num_articles</th>
      <th>population</th>
      <th>num_high_quality_articles</th>
      <th>articles_per_population</th>
      <th>high_qual_article_perc</th>
    </tr>
    <tr>
      <th>country</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Tuvalu</th>
      <td>55</td>
      <td>0.01</td>
      <td>5.0</td>
      <td>5500.000000</td>
      <td>0.090909</td>
    </tr>
    <tr>
      <th>Nauru</th>
      <td>53</td>
      <td>0.01</td>
      <td>0.0</td>
      <td>5300.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>San Marino</th>
      <td>82</td>
      <td>0.03</td>
      <td>0.0</td>
      <td>2733.333333</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Monaco</th>
      <td>40</td>
      <td>0.04</td>
      <td>0.0</td>
      <td>1000.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Liechtenstein</th>
      <td>29</td>
      <td>0.04</td>
      <td>0.0</td>
      <td>725.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Tonga</th>
      <td>63</td>
      <td>0.10</td>
      <td>1.0</td>
      <td>630.000000</td>
      <td>0.015873</td>
    </tr>
    <tr>
      <th>Marshall Islands</th>
      <td>37</td>
      <td>0.06</td>
      <td>0.0</td>
      <td>616.666667</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Iceland</th>
      <td>206</td>
      <td>0.40</td>
      <td>2.0</td>
      <td>515.000000</td>
      <td>0.009709</td>
    </tr>
    <tr>
      <th>Andorra</th>
      <td>34</td>
      <td>0.08</td>
      <td>0.0</td>
      <td>425.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Federated States of Micronesia</th>
      <td>38</td>
      <td>0.10</td>
      <td>0.0</td>
      <td>380.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
dataset.sort_values(by = 'articles_per_population',ascending = True)[0:10]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>num_articles</th>
      <th>population</th>
      <th>num_high_quality_articles</th>
      <th>articles_per_population</th>
      <th>high_qual_article_perc</th>
    </tr>
    <tr>
      <th>country</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>India</th>
      <td>986</td>
      <td>1371.3</td>
      <td>14.0</td>
      <td>0.719026</td>
      <td>0.014199</td>
    </tr>
    <tr>
      <th>Indonesia</th>
      <td>214</td>
      <td>265.2</td>
      <td>8.0</td>
      <td>0.806938</td>
      <td>0.037383</td>
    </tr>
    <tr>
      <th>China</th>
      <td>1135</td>
      <td>1393.8</td>
      <td>33.0</td>
      <td>0.814321</td>
      <td>0.029075</td>
    </tr>
    <tr>
      <th>Uzbekistan</th>
      <td>29</td>
      <td>32.9</td>
      <td>1.0</td>
      <td>0.881459</td>
      <td>0.034483</td>
    </tr>
    <tr>
      <th>Ethiopia</th>
      <td>105</td>
      <td>107.5</td>
      <td>1.0</td>
      <td>0.976744</td>
      <td>0.009524</td>
    </tr>
    <tr>
      <th>Zambia</th>
      <td>25</td>
      <td>17.7</td>
      <td>0.0</td>
      <td>1.412429</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Korea, North</th>
      <td>39</td>
      <td>25.6</td>
      <td>7.0</td>
      <td>1.523438</td>
      <td>0.179487</td>
    </tr>
    <tr>
      <th>Thailand</th>
      <td>112</td>
      <td>66.2</td>
      <td>3.0</td>
      <td>1.691843</td>
      <td>0.026786</td>
    </tr>
    <tr>
      <th>Bangladesh</th>
      <td>323</td>
      <td>166.4</td>
      <td>3.0</td>
      <td>1.941106</td>
      <td>0.009288</td>
    </tr>
    <tr>
      <th>Mozambique</th>
      <td>60</td>
      <td>30.5</td>
      <td>0.0</td>
      <td>1.967213</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



And lastly, we look at the top and bottom countries by high quality article percentage. North Korea has the highest percentage at approximately 18% while Tanzania has the lowest at around .2%. Note that there are some countries that have been removed due to not having any high quality articles. The full list of these countries is at the end.


```python
dataset.sort_values(by = 'high_qual_article_perc',ascending = False)[0:10]

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>num_articles</th>
      <th>population</th>
      <th>num_high_quality_articles</th>
      <th>articles_per_population</th>
      <th>high_qual_article_perc</th>
    </tr>
    <tr>
      <th>country</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Korea, North</th>
      <td>39</td>
      <td>25.60</td>
      <td>7.0</td>
      <td>1.523438</td>
      <td>0.179487</td>
    </tr>
    <tr>
      <th>Saudi Arabia</th>
      <td>119</td>
      <td>33.40</td>
      <td>16.0</td>
      <td>3.562874</td>
      <td>0.134454</td>
    </tr>
    <tr>
      <th>Central African Republic</th>
      <td>68</td>
      <td>4.70</td>
      <td>8.0</td>
      <td>14.468085</td>
      <td>0.117647</td>
    </tr>
    <tr>
      <th>Romania</th>
      <td>348</td>
      <td>19.50</td>
      <td>40.0</td>
      <td>17.846154</td>
      <td>0.114943</td>
    </tr>
    <tr>
      <th>Mauritania</th>
      <td>52</td>
      <td>4.50</td>
      <td>5.0</td>
      <td>11.555556</td>
      <td>0.096154</td>
    </tr>
    <tr>
      <th>Bhutan</th>
      <td>33</td>
      <td>0.80</td>
      <td>3.0</td>
      <td>41.250000</td>
      <td>0.090909</td>
    </tr>
    <tr>
      <th>Tuvalu</th>
      <td>55</td>
      <td>0.01</td>
      <td>5.0</td>
      <td>5500.000000</td>
      <td>0.090909</td>
    </tr>
    <tr>
      <th>Dominica</th>
      <td>12</td>
      <td>0.07</td>
      <td>1.0</td>
      <td>171.428571</td>
      <td>0.083333</td>
    </tr>
    <tr>
      <th>United States</th>
      <td>1092</td>
      <td>328.00</td>
      <td>82.0</td>
      <td>3.329268</td>
      <td>0.075092</td>
    </tr>
    <tr>
      <th>Benin</th>
      <td>94</td>
      <td>11.50</td>
      <td>7.0</td>
      <td>8.173913</td>
      <td>0.074468</td>
    </tr>
  </tbody>
</table>
</div>




```python
#dataset.sort_values(by = 'high_qual_article_perc',ascending = True)[0:10]
dataset[dataset['high_qual_article_perc']>0].sort_values(by = 'high_qual_article_perc',ascending = True)[0:10]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>num_articles</th>
      <th>population</th>
      <th>num_high_quality_articles</th>
      <th>articles_per_population</th>
      <th>high_qual_article_perc</th>
    </tr>
    <tr>
      <th>country</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Tanzania</th>
      <td>408</td>
      <td>59.1</td>
      <td>1.0</td>
      <td>6.903553</td>
      <td>0.002451</td>
    </tr>
    <tr>
      <th>Peru</th>
      <td>354</td>
      <td>32.2</td>
      <td>1.0</td>
      <td>10.993789</td>
      <td>0.002825</td>
    </tr>
    <tr>
      <th>Lithuania</th>
      <td>248</td>
      <td>2.8</td>
      <td>1.0</td>
      <td>88.571429</td>
      <td>0.004032</td>
    </tr>
    <tr>
      <th>Nigeria</th>
      <td>682</td>
      <td>195.9</td>
      <td>3.0</td>
      <td>3.481368</td>
      <td>0.004399</td>
    </tr>
    <tr>
      <th>Morocco</th>
      <td>208</td>
      <td>35.2</td>
      <td>1.0</td>
      <td>5.909091</td>
      <td>0.004808</td>
    </tr>
    <tr>
      <th>Fiji</th>
      <td>199</td>
      <td>0.9</td>
      <td>1.0</td>
      <td>221.111111</td>
      <td>0.005025</td>
    </tr>
    <tr>
      <th>Bolivia</th>
      <td>187</td>
      <td>11.3</td>
      <td>1.0</td>
      <td>16.548673</td>
      <td>0.005348</td>
    </tr>
    <tr>
      <th>Brazil</th>
      <td>551</td>
      <td>209.4</td>
      <td>3.0</td>
      <td>2.631328</td>
      <td>0.005445</td>
    </tr>
    <tr>
      <th>Luxembourg</th>
      <td>180</td>
      <td>0.6</td>
      <td>1.0</td>
      <td>300.000000</td>
      <td>0.005556</td>
    </tr>
    <tr>
      <th>Sierra Leone</th>
      <td>166</td>
      <td>7.7</td>
      <td>1.0</td>
      <td>21.558442</td>
      <td>0.006024</td>
    </tr>
  </tbody>
</table>
</div>



Countries with 0 high quality articles:


```python
dataset[dataset['high_qual_article_perc']==0].index
```




    Index(['Andorra', 'Angola', 'Antigua and Barbuda', 'Bahamas', 'Barbados',
           'Belgium', 'Belize', 'Cameroon', 'Cape Verde', 'Comoros', 'Costa Rica',
           'Djibouti', 'Federated States of Micronesia', 'Finland', 'Guyana',
           'Kazakhstan', 'Kiribati', 'Lesotho', 'Liechtenstein', 'Macedonia',
           'Malta', 'Marshall Islands', 'Moldova', 'Monaco', 'Mozambique', 'Nauru',
           'Nepal', 'San Marino', 'Sao Tome and Principe', 'Seychelles',
           'Slovakia', 'Solomon Islands', 'Switzerland', 'Tunisia', 'Turkmenistan',
           'Uganda', 'Zambia'],
          dtype='object', name='country')




```python
import matplotlib.pyplot as plt
plt.scatter(np.log(dataset['high_qual_article_perc']+.0001),
            np.log(dataset['articles_per_population']),
            c='r',
            s=1
           )
plt.show()
```


![png](output_34_0.png)


# Learnings 

From this analysis, we expected to see varying amounts of both coverage and quality articles as we look at different countries. While I expected there to be better coverage and quality for more developed nations, this did not appear to be the case. It is true that are discrepancies between nations, in large part due to the extreme differences in population between countries. There are many country-specific factors we have not included in this analysis that may help illustrate the trend including education, access to internet, wikipedia popularity, government internet regulations and more.
