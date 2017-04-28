---
layout: post
title: Billboard Top 100 Project
category: Projects
disqus: disabled
excerpt: >-
  The billboard.csv file contains a list of 317 songs that appeared on the
  billboard top 100 list in the year 2000.
published: true
---

##  The billboard.csv file contains a list of 317 songs that appeared on the billboard top 100 list in the year 2000. We have 76 weeks of data on each track, with information on when the track first entered the list and when the track peaked.###


Step 1: Exploring the data.
The first objective here is to load the csv file into a pandas DataFrame so that we can look at the file in a more comfortable table format. Once we've done that we can look at the column headers and as well as the data types of each column, to look for any issues or concerns.

Step 2: Cleaning the data.
Once we're sufficiently comfortable with data structure and format we want to go ahead and implement any changes or modifications required for further calculations or visualizations.

In the Billboard dataset no song lasted on the charts for 76 weeks so there is no data in any of the columns from week 64 and on. We removed the columns that contained no data (i.e. the entire column was filled with NaN). As an additional note, the 1st week column is of type int and the date columns are objects which we will converted to float and datetime types respectively.

{% highlight ruby %}
billboard_data['date.entered'] = pd.to_datetime(billboard_data['date.entered'])
billboard_data['date.peaked'] = pd.to_datetime(billboard_data['date.peaked'])
# convert the x1st.week column from int to float
billboard_data['x1st.week'] = billboard_data['x1st.week'].astype(float)
# print billboard_data.dtypes # confirm both columns were changed properly
billboard_data = billboard_data.dropna(axis=1,how='all')
{% endhighlight %}

Step 3: Visualizing the data.
The next thing that we can do that's helpful is to visually explore the dataset and by pulling together some quick rudimentary graphs.

For example, we quickly pulled together a new column that's the difference between days.entered and days.peaked and plotted a histogram. This gives us a base level of understanding on how long it takes a track to reach the "peak" ranking on the top 100 list.

<img src="/images/fulls/days_to_peak.png" class='fit-image'>

We also looked at what artists have the lowest mean ranking.

<img src="/images/fulls/artist_mean_top100_vert.png" class="fit image">

Step 4: Creating a Problem Statement.
I wanted to pull together some information that would allow us to answer the following question: What genre of music should I choose to write in the year 2000 if I want maximize the time spent in the top 25 on Billboard's top 100 dataset?

Step 5: Brainstorm the Approach.

* Read in CSV data to a Pandas DataFrame
* Clean the CSV data
    * Modify the data types for the appropriate columns
    * Remove the columns where all the values are null
* Create a new dataframe where we pivot the numerical values to a new column and keep the id values as is using the melt function
* Create a pivot table that indexes on the genre column and look at the mean rank by genre
* Export Data into Tableau
* Filter by songs that occured in the top 25 and count the number of occurances in the top 25
* Pull together a visual representation of the average number of weeks a song appears in the top 25 by genre

Here's a packed bubbles plot. Each bubble represents the occurance of a track, with a top 25 ranking broken out by genre type. From here we can quickly see that the Rock genre dominated the top 25 rankings in 2000.
<img src="/images/fulls/billboard_data_top_25_occur.png" class="fit image">

Here we pulled together a box & whisker plot. Each datapoint represents the count of the number of weeks a specific track appeared in the top 25. We can conclude that on average, Rock songs spend the most time in the top 25, with an average of 14.69 weeks spent in the top 25. Latin and Rap are next in line with an average of 11.5 weeks and 10.56 weeks respectively.
<img src="/images/fulls/billboard_data_top25_whisk.png" class="fit image">
