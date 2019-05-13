# RandomForestClassifier_ChubMackerel
I used a random forest classifier to predict the origin of a mackerel (fish) species from samples taken throughout the Atlantic Ocean. 
Predictor features describe the shape of ear bones in each fish.

![fig2](https://github.com/njermain/RandomForestClassifier_ChubMackerel/blob/master/Sampling_Design_v4.jpg)

## The Task

I was trying to show the utility of a type of analysis that groups the origin of fish samples from a particular species given the shape 
of the fish’s ear bone. The basic concept is that fish in distinct groups for a specific species, say Cod, have a unique ear bone shape 
that can be used to identify which geographic region they came from. I wanted to show how well the features I engineered predicted the 
geographic origin where each sample came from; in essence, does the technique have suitable predictive capacity?


## Data Collection

You might ask, how do you quantify the shape of the ear bones? I used a discrete Wavelet transformation (similar to the commonly 
used Fourier transformation) to describe the shape of the outline of each bone (Figure 1) in terms of cosine waves. The transformation 
gave me 64 coefficients to describe the morphological nuances of each fish’s ear bone, these are our features.

![fig2](https://github.com/njermain/RandomForestClassifier_ChubMackerel/blob/master/Rfish2.JPG)

Figure 1: Ear bone and the derived outline using a Wavelet transformation

## Data Processing

I had unequal sampling among demographic groups; without getting into the biological details, I needed to isolate the impact of 
geographic origin on bone shape without demographic details (i.e. length, age, etc.) contributing to the relationship. I used repeated
ANCOVAs to eliminate features where demographic variables significantly covaried with geographic region, and applied a Bonferonni 
adjustment to minimize the buildup of type one error from repeated analyses.

My response variable, geographic region, was sampled unequally (Figure 2). 

![fig2](https://github.com/njermain/RandomForestClassifier_ChubMackerel/blob/master/Rfish3.JPG)

Figure 2: Sample size for each geographic region (1-Gulf of Mexico, 2-West Atlantic, 0-East Atlantic)

I wanted to show that the features I engineered can predict the origin of fish from multiple different regions and I wanted 
to minimize the impact of the variation in sample size on model prediction. To do this, I randomly undersampled the most common class 
(Gulf, in blue).

![fig2](https://github.com/njermain/RandomForestClassifier_ChubMackerel/blob/master/Rfish4.JPG)

Figure 3: Sample size for each geographic region after undersampling

## Modeling

I used a random forest classifier to predict the region the sample came from given the features describing bone shape. First, I
determined the optimal hyperparameter values:

**max_features**: the maximum number of features to consider at each split

**max_depth**: the maximum number of splits in any tree

**min_samples_split**: the minimum number of samples required to split a node

**min_samples_leaf**: the minimum number of samples required at each leaf node

**bootstrap**: whether the data set is bootstrapped or whether the whole dataset is used for each tree

**criterion**: the function used to assess the quality of each split

The sci-kit learn module has a handy method “GridSearchCV” to find optimal hyperparameter values through cross-validation.
I used k-fold cross-validation with 5 folds.

The model predicted the geographic origin of each sample in the test set with 89% accuracy. My prediction accuracy was higher than 
studies that classified the origin of similar fish species. This exercise was limited by a small sample size; the species I studied 
are not caught often.

A classification matrix gives us insight into how the model predictions related to the observed classes.

![fig2](https://github.com/njermain/RandomForestClassifier_ChubMackerel/blob/master/Rfish5.JPG)

The random forest model predicted fish samples from the Gulf of Mexico with much greater accuracy than those from the East Atlantic and West Atlantic. This suggests bone shape is more unique in the Gulf of Mexico than other regions.

This exercise shows the value of machine learning concepts in fisheries science, and their ability to predict the origin of fish samples using the technique I suggest. Given the small sample size, I think the features I engineered offer 
a strong predictive capacity.






