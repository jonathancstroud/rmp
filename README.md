# MDST Ratings Analysis Challenge

Here, we'll be sharing tutorial and demo code from the MDST Ratings
Analysis Challenge.

[Competition Page](https://inclass.kaggle.com/c/mdst-ratings-analysis)


## Scripts

ridge.py - Ridge regression baseline code. Uses regularized sum of
bigram features to predict quality. Leaderboard RMSE: 1.8431

transformdata.py - Cleans up categorical variables to be used with
other learners. Written by Alexander Zaitzeff


## Flux

template_ridge.pbs - Use this PBS script to run the ridge regression
code remotely on Flux. You'll need your account linked to our Flux
allocation (contact Jonathan at stroud@umich.edu). Make sure to change
this line at the top:

```
#PBS -M youruniqname@umich.edu
```

to match your actual uniqname. To run:

```
qsub ridge.pbs
```

## Data

For a complete description of the dataset, see the kaggle in-class
webpage: 

[Data Description](https://inclass.kaggle.com/c/mdst-ratings-analysis/data)