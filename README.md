This project uses the same data files as k-means and k-metoids so it has a large number of extreme
outliers. 
This makes it hard to find the best epsilon and min_sample values even with the data
log transformed and scaled.
The second file has methods in it to try to find the best combination of epsilons and min_sample options.
The effect of the outliers is extreme in this example. KMeans and KMetoids both work well with 4 clusters. DBSCAN has
11 clusters and still has a large number of outliers identified as 'noise'. Going with smaller values creates even larger
'noise' values
