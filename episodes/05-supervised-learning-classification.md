[Go to main page](../README.md)

# Supervised Learning : classification

Supervised learning is the branch of Machine Learning (ML) that involves predicting labels, such as 'Survived' or 'Not'. Such models learn from labelled data, which is data that includes whether a passenger survived (called "model training"), and then predict on unlabeled data.

These are generally called train and test sets because
- You want to build a model that learns patterns in the training set, and
- You then use the model to make predictions on the test set.

We can then calculate the percentage that you got correct: this is known as the accuracy of your model.

## Table of Content <a id="toc"></a>

 1. [our first classifier : Decision trees](#dt1)
 2. [The classifiation pipeline](#pipeline)
 2.1. [Leakage](#leak)
 
[back to ToC](#toc)

## How To Start with Supervised Learning

As you might already know, a good way to approach supervised learning is the following:
- Perform an Exploratory Data Analysis (EDA) on your data set;
- Build a quick and dirty model, or a baseline model, which can serve as a comparison against later models that you will build;
- Iterate this process. You will do more EDA and build another model;
- Engineer features: take the features that you already have and combine them or extract more information from them to eventually come to the last point, which is
- Get a model that performs better.

A common practice in all supervised learning is the construction and use of the **train- and test- datasets**. This process takes all of the input randomly splits into the two datasets (training and test); the ratio of the split is usually up to the researcher, and can be anything: 80/20, 70/30, 60/40...

There are various classifiers available:

- **Decision Trees** – These are organized in the form of sets of questions and answers in the tree structure.
- **Naive Bayes Classifiers** – A probabilistic machine learning model that is used for classification.
- **K-NN Classifiers** – Based on the similarity measures like distance, it classifies new cases.
- **Support Vector Machines** – It is a non-probabilistic binary linear classifier that builds a model to classify a case into one of the two categories.


We will mostly focus on decisions trees for now, to explore and experiment core concepts of the classification pipeline in Machine Learning. 

Once these concepts are well understood, it is actually relatively easy to apply them to more and more algorithms.

[back to ToC](#toc)

## 1. our first classifier : Decision trees <a id='dt1'></a>

It is a type of supervised learning algorithm. We use it for classification problems. It works for both types of input and output variables. In this technique, we split the population into two or more homogeneous sets. Moreover, it is based on the most significant splitter/differentiator in input variables.

The Decision Tree is a powerful non-linear classifier. A Decision Tree makes use of a tree-like structure to generate relationship among the various features and potential outcomes. It makes use of branching decisions as its core structure.

There are two types of decision trees:
- **Categorical (classification)** Variable Decision Tree: Decision Tree which has a categorical target variable.
- **Continuous (Regression)** Variable Decision Tree: Decision Tree has a continuous target variable.

Regression trees are used when the dependent variable is continuous while classification trees are used when the dependent variable is categorical. In continuous, a value obtained is a mean response of observation. In classification, a value obtained by a terminal node is a mode of observations.

The main advantages of the decision tree is :
 * **no need to scale your data**
 * ability to do **non-linear** classification
 * resulting models are **easy to interpret**


Here, we will use the `rpart` and the `rpart.plot` package in order to produce and visualize a decision tree. First of all, we'll create the train and test datasets using a 70/30 ratio and a fixed seed so that we can reproduce the results.

```r
# split into training and test subsets
library(caret)
set.seed(1000)
inTraining <- createDataPartition(breastCancerData$Diagnosis, p = .70, list = FALSE)
breastCancerData.train <- breastCancerDataNoID[ inTraining,]
breastCancerData.test  <- breastCancerDataNoID[-inTraining,]

```

Now, we will load the library and create our model. We would like to create a model that predicts the `Diagnosis` based on the mean of the radius and the area, as well as the SE of the texture. For ths reason we'll use the notation of `myFormula <- Diagnosis ~ Radius.Mean + Area.Mean + Texture.SE`. If we wanted to create a prediction model based on all variables, we will have used `myFormula <- Diagnosis ~ .` instead. 

The decision tree algorithm comes with a number of parameters which reflect the following aspects of the model:
- `minsplit`: the minimum number of instances in a node so that it is split
- `minbucket`: the minimum allowed number of instances in each leaf of the tree
- `maxdepth`: the maximum depth of the tree
- `cp`: parameter that controls the complexity for a split and is set intuitively (the larger its value, the more probable to apply pruning to the tree)

These parameters of the methods which are not set directly from our data itself are called **hyper-parameters**.

```r
library(rpart)
library(rpart.plot)
myFormula <- Diagnosis ~ Radius.Mean + Area.Mean + Texture.SE

breastCancerData.model <- rpart(myFormula,
                                method = "class",
                                data = breastCancerData.train,
                                minsplit = 10,
                                minbucket = 1,
                                maxdepth = 3,
                                cp = -1)

print(breastCancerData.model$cptable)
rpart.plot(breastCancerData.model)

```

We see the following output and a figure:

![Full decision tree](https://raw.githubusercontent.com/fpsom/2021-06-ml-elixir-fr/main/static/images/decisionTreeFull.png "Full decision tree")

```
           CP nsplit rel error    xerror       xstd
1  0.69127517      0 1.0000000 1.0000000 0.06484708
2  0.02013423      1 0.3087248 0.3087248 0.04281476
3  0.00000000      2 0.2885906 0.3154362 0.04321630
4 -1.00000000      6 0.2885906 0.3154362 0.04321630
```

Here, you can see that different values of `CP` have been explored. 
Interestingly, each value tested are associated with :
 * `rel error` : relative error
 * `xerror` : cross-validation error (`rpart` does a 10-fold cross validation without telling you, how nice!)

As we can observe, the value with the lowest relative error (CP: -1), does not correspond to the value with the lowest cross-validation error (CP:0.2013423)

_Question_: **Why is that? Which one should we try to optimize?**

<br>

---

<br>

Let's select the tree with the minimum prediction error:

```r
errorType = "xerror" # would you choose "rel error" or "xerror" ?
opt <- which.min(breastCancerData.model$cptable[, errorType])
cp <- breastCancerData.model$cptable[opt, "CP"]
# prune tree
breastCancerData.pruned.model <- prune(breastCancerData.model, cp = cp)
# plot tree
rpart.plot(breastCancerData.pruned.model)

table(predict(breastCancerData.pruned.model, type="class"), breastCancerData.train$Diagnosis)

```

The output now is the following Confusion Matrix and pruned tree:

```
    B    M
B  245  34
M   9   109
```

![Pruned decision tree](https://raw.githubusercontent.com/fpsom/2021-06-ml-elixir-fr/main/static/images/decisionTreePruned.png "Pruned decision tree")

_Question: **What does the above "Confusion Matrix" tells you?**_


So when it increases the CP parameter's effect will reduce the performance on the train set (`rel error`), but limit the amount of error done on new data (`xerror`).
In other words, it can help avoid **overfitting** of the model. In a sense it is a **regularization parameter**.

Regularization is a double edged sword:
 * **too little regularization** : overfitting on the train set, bad performance on new data (eg. test set)
 * **too much regularization** : bad performance overall

Anyway, now that we have a model, we should check how the prediction works in our test dataset.


```r
## make prediction
BreastCancer_pred <- predict(breastCancerData.pruned.model, newdata = breastCancerData.test, type="class")
plot(BreastCancer_pred ~ Diagnosis, data = breastCancerData.test,
     xlab = "Observed",
     ylab = "Prediction")
table(BreastCancer_pred, breastCancerData.test$Diagnosis)
```

The new Confusion Matrix is the following:

```
BreastCancer_pred   B   M
                B 104  11
                M   3  52
```

![Prediction Plot](https://raw.githubusercontent.com/fpsom/2021-06-ml-elixir-fr/main/static/images/predictionPlot.png "Prediction Plot")

| **Exercises**  |   |
|--------|----------|
| 1 | Can we improve the above model? What are the key hyperparameters that have the most impact?|
| 2 | We have been using only some of the variables in our model. What is the impact of using all variables / features for our prediction? Is this a good or a bad plan?|


[back to ToC](#toc)

## 2. "The classiciation pipeline" <a id='pipeline'></a>

Now that we have experimented a little with a type model (decision tree), we will be able to discuss important aspects of classification routines in Machine Learning: what are the aspects that are common to most approaches ? what to strive for ? which pitfall to avoid ?

### 2.1. Leakage - the insidious foe <a id='leak'></a>

Consider the following code, which explores different values for the parameter `maxdepth`:

```r

v.maxdepth = 1:10
v.xerror = c()
v.predError = c()

for(maxdepth in v.maxdepth){

  breastCancerData.model <- rpart( 'Diagnosis ~ . ' ,
                                method = "class",
                                data = breastCancerData.train,
                                minsplit = 10,
                                minbucket = 1,
                                maxdepth = maxdepth,
                                cp = -1)


  opt <- which.min(breastCancerData.model$cptable[, "xerror"])

  cp <- breastCancerData.model$cptable[opt, "CP"]
  breastCancerData.pruned.model <- prune(breastCancerData.model, cp = cp)


  xerror = min( breastCancerData.model$cptable[, "xerror"] )
  predError = mean( predict(breastCancerData.pruned.model, newdata = breastCancerData.test, type="class") != breastCancerData.test$Diagnosis )

  v.xerror = c(v.xerror , xerror)
  v.predError = c(v.predError , predError)
}


plot(v.maxdepth , v.xerror , type='l' , lwd=2, 
	ylim=c(0.0,1.0) , xlab='maxdepth' , ylab='error')
lines(v.maxdepth , v.predError  , col='red', lwd=2)

legend('topright', c('cross-validation error' , 'prediction error') , lwd=2, col=c('black','red'))

m = which.min(v.xerror)
points(v.maxdepth[m] , v.xerror[m]  , col='black', lwd=2)
m = which.min(v.predError)
points(v.maxdepth[m] , v.predError[m]  , col='red', lwd=2)

```

![maxdepth exploration](https://raw.githubusercontent.com/fpsom/2021-06-ml-elixir-fr/main/static/images/maxdepthExploration.png "maxdepth exploration")

Where the little circle mark the points where a minimum is reached.

_Question:_**Which `maxdepth` should I choose here for my final model?**

??? done "answer"

  test 

### 2.2. 


### Random Forests

Random Forests is an ensemble learning technique, which essentially constructs multiple decision trees. Each tree is trained with a random sample of the training dataset and on a randomly chosen subspace. The final prediction result is derived from the predictions of all individual trees, with mean (for regression) or majority voting (for classification). The advantage is that it has better performance and is less likely to overfit than a single decision tree; however it has lower interpretability.

There are two main libraries in R that provide the functionality for Random Forest creation; the `randomForest` and the `party: cforest()`.

Package `randomForest`
- very fast
- cannot handle data with missing values
- a limit of 32 to the maximum number of levels of each categorical attribute
- extensions: extendedForest, gradientForest


Package `party: cforest()`
- not limited to the above maximum levels
- slow
- needs more memory

In this exercise, we will be using the `randomForest`. First, let's train the model:

```r
library(randomForest)
set.seed(1000)
rf <- randomForest(Diagnosis ~ ., data = breastCancerData.train,
                   ntree=100,
                   proximity=T)

table(predict(rf), breastCancerData.train$Diagnosis)
```

The output is the following:

```
   B   M
B 249  12
M   5 131
```

We can also investigate the content of the model:

```r
print(rf)
```

The output shows the individual components and internal parameters of the Random Forest model.

```
Call:
 randomForest(formula = Diagnosis ~ ., data = breastCancerData.train,      ntree = 100, proximity = T)
               Type of random forest: classification
                     Number of trees: 100
No. of variables tried at each split: 5

        OOB estimate of  error rate: 4.28%
Confusion matrix:
    B   M class.error
B 249   5  0.01968504
M  12 131  0.08391608
```

We can view the overall performance of the model here:

```r
plot(rf, main = "")
```

![Error rate plot for the Random Forest model](https://raw.githubusercontent.com/fpsom/2021-06-ml-elixir-fr/main/static/images/error-rate-rf.png "Error rate plot for the Random Forest model")

We can also review which of the variables has the highest "importance" (i.e. impact to the performance of the model):

```r
importance(rf)

varImpPlot(rf)
```

The output is the table and the figure below:

```
ID                             1.0244803
Radius.Mean                    7.8983552
Texture.Mean                   1.9614134
Perimeter.Mean                 9.3502914
Area.Mean                      7.3438007
Smoothness.Mean                0.7228277
Compactness.Mean               2.6595043
Concavity.Mean                11.2341661
Concave.Points.Mean           18.5940046
Symmetry.Mean                  0.8989458
Fractal.Dimension.Mean         0.7465322
Radius.SE                      3.1941672
Texture.SE                     0.6363906
Perimeter.SE                   2.4672730
Area.SE                        5.3446273
Smoothness.SE                  0.6089522
Compactness.SE                 0.7785777
Concavity.SE                   0.5576146
Concave.Points.SE              1.0314107
Symmetry.SE                    0.8839428
Fractal.Dimension.SE           0.6475348
Radius.Worst                  18.2035365
Texture.Worst                  3.2765864
Perimeter.Worst               25.3605679
Area.Worst                    17.1063000
Smoothness.Worst               2.1677456
Compactness.Worst              2.9489506
Concavity.Worst                6.0009637
Concave.Points.Worst          25.6081497
Symmetry.Worst                 2.1507714
Fractal.Dimension.Worst        1.1498020
```

![Importance of the individual variables](https://raw.githubusercontent.com/fpsom/2021-06-ml-elixir-fr/main/static/images/importance-variables.png "Importance of the individual variables")

Let's try to do a prediction of the `Diagnosis` for the test set, using the new model. The margin of a data point is as the proportion of votes for the correct class minus maximum proportion of votes for other classes. Positive margin means correct classification.

```r
BreastCancer_pred_RD <- predict(rf, newdata = breastCancerData.test)
table(BreastCancer_pred_RD, breastCancerData.test$Diagnosis)

plot(margin(rf, breastCancerData.test$Diagnosis))
```

The output is the table and figure below:

```
BreastCancer_pred_RD   B   M
                   B 101   6
                   M   2  63
```

![Margin plot for the Random Forest](https://raw.githubusercontent.com/fpsom/2021-06-ml-elixir-fr/main/static/images/margin-rf.png "Margin plot for the Random Forest")

Feature selection: We can evaluate the prediction performance of models with reduced numbers of variables that are ranked by their importance.

```r
result <- rfcv(breastCancerData.train, breastCancerData.train$Diagnosis, cv.fold=3)
with(result, plot(n.var, error.cv, log="x", type="o", lwd=2))
```

![Random Forest Cross-Validation for feature selection](https://raw.githubusercontent.com/fpsom/2021-06-ml-elixir-fr/main/static/images/rfcv.png "Random Forest Cross-Valdidation for feature selection")

| **Exercises**  |   |
|--------|----------|
| 1 | ToDo |
| 2 | ToDo |

