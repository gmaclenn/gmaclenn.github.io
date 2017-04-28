---
layout: post
title: Principal Component Analysis & Clustering with Airport Delay Data
category: Projects
disqus: disabled
excerpt: In this project, I looked at three different datasets related to airport operations.
---


# Airport Delays + Cluster Analysis

#### Project Summary

In this project, I looked at three different datasets related to airport operations. These include a dataset detailing the arrival and departure delays/diversions by airport, a dataset that provides metrics related to arrivals and departures for each airport, and a dataset that details names and characteristics for each airport code. In the end I will see if there are specific characteristics of airports that lead to increased delays.

#### Read in the CSV files


```python
df_raw = pd.read_csv("../assets/airport_cancellations.csv")
cancellations_df = pd.read_csv('../assets/Airport_operations.csv')
airports_df = pd.read_csv('../assets/airports.csv')
```

#### Convert all column names to lower case


```python
def columns_to_lower(df):
    """lowercase all column names in a dataframe"""
    df.columns = [i.lower() for i in df.columns]

columns_to_lower(df)
columns_to_lower(airports_df)
```

#### Join all three csv files into one dataframe


```python
combined_df = pd.merge(cancellations_df, airports_df, left_on='airport', right_on='locid')
all_data = pd.merge(combined_df, df, left_on=['airport', 'year'], right_on=['airport', 'year'])

# combined_df.to_csv('/Users/gmaclenn/jupyter-notebooks/Projects/combined_airports.csv') # export for tableau analysis
# all_data.to_csv('/Users/gmaclenn/jupyter-notebooks/Projects/all_data.csv') # export for tableau analysis
```


#### Exploratory Data Analysis

<img src="/images/fulls/project7-pca-analysis-greg_files/project7-pca-analysis-greg_16_0.png" class="fit image"/>


Above we see there's an almost perfect correlation between arrivals and departures. This is not surprising as you should expect the number of planes coming in should equal the number of planes that are leaving. However, this is a perfect example of where PCA can be useful. These are two seperate columns of data that are basically explaining the exact same variances in the data.

Below we can see that the gate departure delays look to have a normal distribution.


<img src="/images/fulls/project7-pca-analysis-greg_files/project7-pca-analysis-greg_18_0.png" class="fit image"/>



The image below shows the gate departure delays within the context of a US map. The size of the bubbles correspond to the total number of departures from a given airport and the red is roughly areas that have average gate departure delays in excess of the median. Where as white is roughly the median value at 12.54 minutes.


<img src="/images/fulls/project7-pca-analysis-greg_files/project7-pca-analysis-greg_20_0.png" class="fit image"/>



### Part 3: Data Mining

#### Drop unnecessary columns


```python
drop_cols = ['key', 'locid', 'ap_name', 'alias', 'facility type', 'faa region', 'boundary data available', 'ap type',
            'county', 'city', 'state', 'latitude', 'longitude', 'airport']
airport_pca_data = all_data.drop(drop_cols, axis=1) # drops columns
airport_pca_data.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>departures for metric computation</th>
      <th>arrivals for metric computation</th>
      <th>percent on-time gate departures</th>
      <th>percent on-time airport departures</th>
      <th>percent on-time gate arrivals</th>
      <th>average_gate_departure_delay</th>
      <th>average_taxi_out_time</th>
      <th>average taxi out delay</th>
      <th>average airport departure delay</th>
      <th>average airborne delay</th>
      <th>average taxi in delay</th>
      <th>average block delay</th>
      <th>average gate arrival delay</th>
      <th>departure cancellations</th>
      <th>arrival cancellations</th>
      <th>departure diversions</th>
      <th>arrival diversions</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2004</td>
      <td>53971</td>
      <td>53818</td>
      <td>0.8030</td>
      <td>0.7809</td>
      <td>0.7921</td>
      <td>10.38</td>
      <td>9.89</td>
      <td>2.43</td>
      <td>12.10</td>
      <td>2.46</td>
      <td>0.83</td>
      <td>2.55</td>
      <td>10.87</td>
      <td>242.0</td>
      <td>235.0</td>
      <td>71.0</td>
      <td>46.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2005</td>
      <td>51829</td>
      <td>51877</td>
      <td>0.8140</td>
      <td>0.7922</td>
      <td>0.8001</td>
      <td>9.60</td>
      <td>9.79</td>
      <td>2.29</td>
      <td>11.20</td>
      <td>2.26</td>
      <td>0.89</td>
      <td>2.34</td>
      <td>10.24</td>
      <td>221.0</td>
      <td>190.0</td>
      <td>61.0</td>
      <td>33.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2006</td>
      <td>49682</td>
      <td>51199</td>
      <td>0.7983</td>
      <td>0.7756</td>
      <td>0.7746</td>
      <td>10.84</td>
      <td>9.89</td>
      <td>2.16</td>
      <td>12.33</td>
      <td>2.12</td>
      <td>0.84</td>
      <td>2.66</td>
      <td>11.82</td>
      <td>392.0</td>
      <td>329.0</td>
      <td>71.0</td>
      <td>124.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2007</td>
      <td>53255</td>
      <td>53611</td>
      <td>0.8005</td>
      <td>0.7704</td>
      <td>0.7647</td>
      <td>11.29</td>
      <td>10.34</td>
      <td>2.40</td>
      <td>12.95</td>
      <td>2.19</td>
      <td>1.29</td>
      <td>3.06</td>
      <td>12.71</td>
      <td>366.0</td>
      <td>304.0</td>
      <td>107.0</td>
      <td>45.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2008</td>
      <td>49589</td>
      <td>49512</td>
      <td>0.8103</td>
      <td>0.7844</td>
      <td>0.7875</td>
      <td>10.79</td>
      <td>10.41</td>
      <td>2.41</td>
      <td>12.32</td>
      <td>1.82</td>
      <td>1.03</td>
      <td>2.79</td>
      <td>11.48</td>
      <td>333.0</td>
      <td>300.0</td>
      <td>79.0</td>
      <td>42.0</td>
    </tr>
  </tbody>
</table>
</div>

#### Find correlations in the data


```python
corrmat = airport_pca_data.corr()
sns.heatmap(corrmat)
```

<img src="/images/fulls/project7-pca-analysis-greg_files/project7-pca-analysis-greg_28_1.png" class="fit image"/>


#### Strong correlations:
* Total Departures & Arrivals
* Total Departures/Arrivals & Departure/Arrival Cancellations and Diversions
* Avg. Gate Departure Delay & Avg. Gate Arrival Delay
* All cancellations/Deversions & Avg. Taxi in Delay
* Departures/Cancellations & themselves

<img src="/images/fulls/project7-pca-analysis-greg_files/project7-pca-analysis-greg_30_0.png" class="fit image"/>


In looking at the correlations prior to performing PCA we can get an idea for what variables are highly correlated with eachother. In this regard we can see variables that are likely to be diminished as they explain the same variance.

#### Prepare the data for PCA


```python
# split the dataframe into X and y values
X, y = airport_pca_data.iloc[:, 1:].values, airport_pca_data.iloc[:, 0].values
```

#### Standardize the data using the StandardScaler


```python

ss = StandardScaler()

X_std = ss.fit_transform(X)
```

#### From the covariance matrix calculate the eigenvalues and eigen vectors


```python
cov_mat = np.cov(X_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
```

```python
print eigen_vals
```

    [  9.17680918e+00   4.16210652e+00   1.21645796e+00   6.50383255e-01
       5.10203215e-01   4.20437718e-01   2.47407511e-01   1.90068175e-01
       1.65777857e-01   1.43298219e-01   6.19936947e-02   5.37682509e-02
       1.49627279e-02   4.86205637e-03   1.71481675e-04   9.58506528e-04
       1.63693067e-03]



```python
eigen_vals
```

```python
tot = sum(eigen_vals) # totals the eigen values
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)] # var_exp ratio is fraction of eigen_val to total sum
cum_var_exp = np.cumsum(var_exp) # calculate the cumulative sum of explained variances
```


```python
import matplotlib.pyplot as plt

plt.bar(range(1,18), var_exp, alpha=0.75, align='center', label='individual explained variance')
plt.step(x=range(1,18), y=cum_var_exp, where='mid', label='cumulative explained variance')
plt.ylim(0, 1.1)
plt.xlabel('Principal components')
plt.ylabel('Explained variance ratio')
plt.legend(loc='best')
print "Here we can see what perentage of the variance is explained by each principal component, since we are sorted"
print "by the magnitude of the eigenvalues"
plt.show()
```

    Here we can see what perentage of the variance is explained by each principal component, since we are sorted
    by the magnitude of the eigenvalues

<img src="/images/fulls/project7-pca-analysis-greg_files/project7-pca-analysis-greg_43_1.png" class="fit image"/>


## What does this tell us?

#### The first two principal components combined explain almost 80% of the variance in the data. The first three components explain 85% of all the variance in the data.


```python
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

pca = PCA(n_components=2) # PCA with 2 primary components
pca_3 = PCA(n_components=3) #PCA with 3 primary components

# fit and transform both PCA models
X_pca = pca.fit_transform(X_std)
X_pca_3 = pca_3.fit_transform(X_std)
```


```python
print X_pca.shape, X_pca_3.shape # (rows, n_components)
```

    (799, 2) (799, 3)



```python
plt.scatter(X_pca.T[0], X_pca.T[1], c='blue')
plt.xlabel('PC1')
plt.ylabel('PC2')
print "Here we're able to see our airport data in our 2-d feature subspace"
print "There's still quite a bit of noise and not a clear delineation in the 2-d feature subspace"
plt.show()
```

    Here we're able to see our airport data in our 2-d feature subspace
    There's still quite a bit of noise and not a clear delineation in the 2-d feature subspace


<img src="/images/fulls/project7-pca-analysis-greg_files/project7-pca-analysis-greg_48_1.png" class="fit image"/>


### We'll see if adding a third principal component allows for better distinction between datapoints


```python
from mpl_toolkits.mplot3d import Axes3D

# initialize figure and 3d projection for the PC3 data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# assign x,y,z coordinates from PC1, PC2 & PC3
xs = X_pca_3.T[0]
ys = X_pca_3.T[1]
zs = X_pca_3.T[2]

# initialize scatter plot and label axes
ax.scatter(xs, ys, zs, alpha=0.4, c='blue')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

# set axes limits
ax.set_xlim3d(-12,12)
ax.set_ylim3d(-10,10)
ax.set_zlim3d(-10,10)

plt.show()
```

<img src="/images/fulls/project7-pca-analysis-greg_files/project7-pca-analysis-greg_50_0.png" class="fit image"/>


#### In my opinion adding PC3 does not help differentiate the data significantly.

## Using KMeans to cluster the Principal Components

### Use the elbow method to approximate the optimal number of clusters


```python
distortions = [] # sum of squared error within the each cluster
for i in range(1, 11):
    km = KMeans(n_clusters=i,
               init='k-means++',
               n_init=10,
               max_iter=300,
               random_state=0)
    km.fit(X_std)
    distortions.append(km.inertia_)

plt.plot(range(1,11), distortions, marker='o', alpha=0.75)
plt.xlabel('Number of clusters')
plt.ylabel('Distortions')
print "As we can see here there's not a clear optimal # of clusters based on the SSE values within each cluster"
plt.show()
```

    As we can see here there's not a clear optimal # of clusters based on the SSE values within each cluster

<img src="/images/fulls/project7-pca-analysis-greg_files/project7-pca-analysis-greg_56_1.png" class="fit image"/>


## We'll choose K=3 but also look at K=2 and K=4

```python
from sklearn.cluster import KMeans

km = KMeans(n_clusters=3,
           init='k-means++',
           n_init=10,
           max_iter=300,
           tol=1e-04,
           random_state=0)

y_km = km.fit_predict(X_pca)
```


```python
plt.scatter(X_pca[y_km==0, 0],
           X_pca[y_km==0, 1],
           c='lightgreen',
           label='Cluster 1')
plt.scatter(X_pca[y_km==1, 0],
           X_pca[y_km==1, 1],
           c='orange',
           label='Cluster 2')
plt.scatter(X_pca[y_km==2, 0],
           X_pca[y_km==2, 1],
           c='lightblue',
           label='Cluster 3')
plt.scatter(X_pca[y_km==3, 0],
           X_pca[y_km==3, 1],
           c='yellow',
           label='Cluster 4')
plt.scatter(km.cluster_centers_[:,0],
           km.cluster_centers_[:,1],
           s=85,
           alpha=0.75,
           marker='o',
           c='black',
           label='Centroids')

plt.legend(loc='best')
plt.xlabel('PC1')
plt.ylabel('PC2')
# plt.savefig('/Users/gmaclenn/Desktop/2_clusters.png', dpi=300)
plt.show()
```

<img src="/images/fulls/project7-pca-analysis-greg_files/project7-pca-analysis-greg_59_0.png" class="fit image"/>

<img src="/images/fulls/project7-pca-analysis-greg_files/project7-pca-analysis-greg_60_0.png" class="fit image"/>

<img src="/images/fulls/project7-pca-analysis-greg_files/project7-pca-analysis-greg_61_0.png" class="fit image"/>


## It looks like we're still missing some of the data that's down in the bottom right corner. Let's see if DBSCAN will help pull that data out.

# Using DBScan as an alternative clustering method


```python
from sklearn.cluster import DBSCAN

dbs = DBSCAN(eps=0.75,
             min_samples=5)

y_dbs = dbs.fit_predict(X_pca)
```


```python
plt.scatter(X_pca[y_dbs==-1, 0],
           X_pca[y_dbs==-1, 1],
           c='lightgreen',
           label='Cluster 1')
plt.scatter(X_pca[y_dbs==0, 0],
           X_pca[y_dbs==0, 1],
           c='orange',
           label='Cluster 2')
plt.scatter(X_pca[y_dbs==1, 0],
           X_pca[y_dbs==1, 1],
           c='lightblue',
           label='Cluster 3')
plt.scatter(X_pca[y_dbs==2, 0],
           X_pca[y_dbs==2, 1],
           c='yellow',
           label='Cluster 4')
plt.scatter(X_pca[y_dbs==3, 0],
           X_pca[y_dbs==3, 1],
           c='pink',
           label='Cluster 5')
plt.scatter(X_pca[y_dbs==4, 0],
           X_pca[y_dbs==4, 1],
           c='purple',
           label='Cluster 6')

plt.legend(loc='best')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.savefig('/Users/gmaclenn/Desktop/dbscan_5_clusters.png', dpi=300)
print "DBSCAN does not appear to do much better, given that there's not a clear separation between points here "
print "There are also a number of bridge points ('noise') that's not allowing for clear seperation."
plt.show()
```

    DBSCAN does not appear to do much better, given that there's not a clear separation between points here
    There are also a number of bridge points ('noise') that's not allowing for clear seperation.

<img src="/images/fulls/project7-pca-analysis-greg_files/project7-pca-analysis-greg_65_1.png" class="fit image"/>

# Executive Summary:

#### * Large % of variables were highly correlated and explained the same variances.

#### * Using PCA we are able to reduce dimensionality.
    * while still maintaining much of the explained variance.

### Possible explanations:

#### * The high correlation between departure delays and arrival delays on a number of the variables would lead me to suggest key areas of determining overall lateness revolved around the origination point of each flight.
    * Any sort of departure delay metric was generally highly correlated with arrival delay.
    * As flights are delayed coming in they're going to delay flights heading out.
        * staff turnaround time.
        * refueling
        * etc.

## Observations from other slides:
* New York airports are some of the worst offenders? What's the connection
* Is there a connection with weather and delays?
* Taxi out & Taxi in is much greater at very large airports.

# Possible Next Steps:

* Flight volume between airports.
    * Is one airport's delay directly causing a delay at other airports?


```python
km_c1 = all_data[y_km==0]
km_c2 = all_data[y_km==1]
km_c3 = all_data[y_km==2]
```


```python
km_c1.to_csv('/Users/gmaclenn/Desktop/km_c1.csv')
km_c2.to_csv('/Users/gmaclenn/Desktop/km_c2.csv')
km_c3.to_csv('/Users/gmaclenn/Desktop/km_c3.csv')
```

<img src="/images/fulls/project7-pca-analysis-greg_files/project7-pca-analysis-greg_72_0.png" class="fit image"/>

<img src="/images/fulls/project7-pca-analysis-greg_files/project7-pca-analysis-greg_73_0.png" class="fit image"/>

<img src="/images/fulls/project7-pca-analysis-greg_files/project7-pca-analysis-greg_74_0.png" class="fit image"/>

<img src="/images/fulls/project7-pca-analysis-greg_files/project7-pca-analysis-greg_75_0.png" class="fit image"/>
