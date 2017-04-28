---
layout: post
title: Iowa Liquor Sales Analysis
category: Projects
disqus: disabled
excerpt: >-
  Using linear regression to predict 2016 liquor sales in Iowa.
published: true
---

## Introduction:
In this hypothetical situation, we are considering changing the Iowa liquor tax rates and want to assess the impact on tax revenues for a given change.

As it stands today, we have no visibility into total liquor sales in the state and are unable to accurately project liquor sales in 2016. Our objective is to project liquor sales for the upcoming year so we can determine what effect a change in the tax rate would have on revenues.

Using data from 2015 liquor sales and Q1'16 liquor sales we'll use linear regression to predict the sales totals for the rest of 2016. Our goal is to develop a model that will allow the tax board to make an informed assesment on modifying the tax rate.

## Exploring the data:
Initially we pulled data from the Iowa State website and into a CSV file. From here we're able to use pandas to modify the file into a DataFrame. Once we've done that we can look at the column headers and as well as the data types of each column, to see if we need to change any data types. A sample of the output below: 

| Column Name | Dtype |
|-------------|-------|
| Invoice/Item Number | object |
|Date                 |    datetime64[ns]|
|Store Number         |             int64|
|Store Name           |            object|
|Address              |            object|
|City                 |            object|
|Zip Code              |           object|
|Store Location         |          object|
|County Number           |        float64|
|Category                 |       float64|
|Vendor Number             |      float64|
|Vendor Name               |       object|
|Item Number               |        int64|
|Pack                      |        int64|
|Bottle Volume (ml)        |        int64|
|State Bottle Cost         |      float64|
|State Bottle Retail       |      float64|
|Bottles Sold              |        int64|
|Sale (Dollars)            |      float64|
|Volume Sold (Liters)      |      float64|
|Volume Sold (Gallons)     |      float64|

## Cleaning the data:
In this particular instance, we needed to remove a few redundant columns, remove "$" symbols and commas from sales data and convert the "Date" column to datetime format. Typically, once ive gone through and cleaned the data, I'll export the values to a new CSV so that we can work directly off of the cleaned dataset for any additonal visualizations or calculations.

Once we're sufficiently comfortable with data structure and format we want to go ahead and implement any changes or modifications required for further calculations or visualizations.

In the Billboard dataset no song lasted on the charts for 76 weeks so there is no data in any of the columns from week 64 and on. We removed the columns that contained no data (i.e. the entire column was filled with NaN). As an additional note, the 1st week column is of type int and the date columns are objects which we will converted to float and datetime types respectively.

## Visualizing the data:
The next thing that we can do that's helpful is to visually explore the dataset and by pulling together some quick rudimentary graphs.

<img src="/images/fulls/2015_vs_q1.png" class="fit image">

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
<img src="/images/fulls/billboard_data_top25_whisk.png">
