# Research exercise: Which model to use when?

<img src="images/screenshot.png" width="500"/>
Note: something like random forest or decision trees can be used for both continuous and categorical problems

## Supervised Models
### Decision trees
<img src="https://user-images.githubusercontent.com/43540613/172396826-af4abd77-261e-45f7-9855-9db13e53db48.png" width="350"/>

Decision Tree algorithm belongs to the family of supervised learning algorithms. Unlike other supervised learning algorithms, the decision tree algorithm can be used for solving regression and classification problems too. The goal of using a Decision Tree is to create a training model that can use to predict the class or value of the target variable by learning simple decision rules inferred from prior data(training data).

### Random Forest
<img src="https://user-images.githubusercontent.com/43540613/172595075-cc931c80-7af9-4bfe-8adf-abcdf7305a79.png" width="400"/>

The random forest algorithm is an extension of the bagging method as it utilizes both bagging and feature randomness to create an uncorrelated forest of decision trees. Feature randomness, also known as feature bagging or "the random subspace method", generates a random subset of features, which ensures low correlation among decision trees. This is a key difference between decision trees and random forests. While decision trees consider all the possible feature splits, random forests only select a subset of those features. By accounting for all the potential variability in the data, we can reduce the risk of overfitting, bias, and overall variance, resulting in more precise predictions.
 
Note: What is an ensemble method?
Ensemble learning methods are made up of a set of classifiers - e.g. decision trees and their predictions are aggregated to identify the most popular result. The most well-known ensemble methods are bagging, also known as bootstrap aggregation, and boosting. In bagging, a random sample of data in a training set is selected with replacement - meaning that the individual data points can be chosen more than once. After several data samples are generated, these models are then trained independently, and depending on the type of task - i.e. regression or classification - the average or majority of those predictions yield a more accurate estimate. This approach is commonly used to reduce variance within a noisy dataset.

Reference: https://www.ibm.com/cloud/learn/random-forest

### XGBoost - eXtreme Gradient Boosting
<img src="https://user-images.githubusercontent.com/43540613/174069237-f11c7d37-7276-494a-98b2-4812a222c4ae.png" width="500"/>

Boosting is an ensemble technique where new models are added to correct the errors made by existing models. Models are added sequentially until no further improvements can be made. A popular example is the AdaBoost algorithm that weights data points that are hard to predict. Gradient boosting is an approach where new models are created that predict the residuals or errors of prior models and then added together to make the final prediction. It is called gradient boosting because it uses a gradient descent algorithm to minimize the loss when adding new models.

This approach supports both regression and classification predictive modeling problems.
One of the most important differences between XG Boost and Random forest is that the XGBoost always gives more importance to functional space when reducing the cost of a model while Random Forest tries to give more preferences to hyperparameters to optimize the model.

Reference: https://medium.com/geekculture/xgboost-versus-random-forest-898e42870f30

### LightGBM - Light Gradient Boosting
<img src="https://user-images.githubusercontent.com/43540613/174068978-0e01ca65-953a-4993-b681-720e7ad54888.png" width="500"/>

Light Gradient Boosted Machine, or LightGBM for short, is an open-source library that provides an efficient and effective implementation of the gradient boosting algorithm. LightGBM extends the gradient boosting algorithm by adding a type of automatic feature selection as well as focusing on boosting examples with larger gradients. This can result in a dramatic speedup of training and improved predictive performance.

As such, LightGBM has become a de facto algorithm for machine learning competitions when working with tabular data for regression and classification predictive modeling tasks. As such, it owns a share of the blame for the increased popularity and wider adoption of gradient boosting methods in general, along with Extreme Gradient Boosting (XGBoost).

Reference: 
- https://machinelearningmastery.com/light-gradient-boosted-machine-lightgbm-ensemble/

### Light GBM or XGBoost?
<img src="https://user-images.githubusercontent.com/43540613/174068302-c3561bd1-7055-4dce-95c9-30dd1b368b54.png" width="500"/>

- In XGBoost, trees grow depth-wise while in LightGBM, trees grow leaf-wise which is the fundamental difference between the two frameworks.
- XGBoost is backed by the volume of its users that results in enriched literature in the form of documentation and resolutions to issues. While LightGBM is yet to reach such a level of documentation.
- Both the algorithms perform similarly in terms of model performance but LightGBM training happens within a fraction of the time required by XGBoost.
- Fast training in LightGBM makes it the go-to choice for machine learning experiments.
- XGBoost requires a lot of resources to train on large amounts of data which makes it an accessible option for most enterprises while LightGBM is lightweight and can be used on modest hardware.
- LightGBM provides the option for passing feature names that are to be treated as categories and handles this issue with ease by splitting on equality. 
- H2O’s implementation of XGBoost provides the above feature as well which is not yet provided by XGBoost’s original library.
- Hyperparameter tuning is extremely important in both algorithms.

References:
- https://neptune.ai/blog/xgboost-vs-lightgbm
- https://www.analyticsvidhya.com/blog/2017/06/which-algorithm-takes-the-crown-light-gbm-vs-xgboost/

### Adaboost
AdaBoost is short for Adaptive Boosting. AdaBoost was the first successful boosting algorithm developed for binary classification. Also, it is the best starting point for understanding boosting algorithms. It is adaptive in the sense that subsequent classifiers built are tweaked in favour of those instances misclassified by previous classifiers. It is sensitive to noisy data and outliers. 

AdaBoost uses multiple iterations to generate a single composite strong learner. It creates a strong learner by iteratively adding weak learners. During each phase of training, a new weak learner is added to the ensemble, and a weighting vector is adjusted to focus on examples that were misclassified in previous rounds. The result is a classifier that has higher accuracy than the weak learner classifiers.

Reference: https://www.mygreatlearning.com/blog/xgboost-algorithm/

### Logistic Regression
<img src="https://user-images.githubusercontent.com/43540613/172834151-b83706d1-713a-4153-819b-fdbd098dac5e.png" width="500"/>

Logistic regression is a process of modeling the probability of a discrete outcome given an input variable. The most common logistic regression models a binary outcome; something that can take two values such as true/false, yes/no, and so on.

#### Should I use logistic regression or a decision tree?
Decision Trees are non-linear classifiers; they do not require data to be linearly separable. When you are sure that your data set divides into two separable parts, then use a Logistic Regression. If you're not sure, then go with a Decision Tree. A Decision Tree will take care of both.

### Linear Regression
<img src="https://user-images.githubusercontent.com/43540613/173571597-06618223-084e-464e-8bdd-9d9b1502a125.png" width="500"/>

Linear regression attempts to model the relationship between two variables by fitting a linear equation to observed data. One variable is considered to be an explanatory variable, and the other is considered to be a dependent variable. For example, a modeler might want to relate the weights of individuals to their heights using a linear regression model. Before attempting to fit a linear model to observed data, a modeler should first determine whether or not there is a relationship between the variables of interest. This does not necessarily imply that one variable causes the other (for example, higher SAT scores do not cause higher college grades), but that there is some significant association between the two variables. A scatterplot can be a helpful tool in determining the strength of the relationship between two variables. If there appears to be no association between the proposed explanatory and dependent variables (i.e., the scatterplot does not indicate any increasing or decreasing trends), then fitting a linear regression model to the data probably will not provide a useful model. A valuable numerical measure of association between two variables is the correlation coefficient, which is a value between -1 and 1 indicating the strength of the association of the observed data for the two variables.

A linear regression line has an equation of the form Y = a + bX, where X is the explanatory variable and Y is the dependent variable. The slope of the line is b, and a is the intercept (the value of y when x = 0).

Remember to check assumptions of linear regression first:
There are four assumptions associated with a linear regression model.
- Linearity: The relationship between X and the mean of Y is linear.
- Homoscedasticity: The variance of residual is the same for any value of X.
- Independence: Observations are independent of each other.
- Normality: For any fixed value of X, Y is normally distributed.

## Unsupervised Models
<img src="https://user-images.githubusercontent.com/43540613/174483309-b4e010d3-deeb-4284-9881-e291f8944913.png" width="500"/>

### Clustering
There are many different types of clustering algorithms

### K-means clustering algorithm
K-means clustering is the most commonly used clustering algorithm. It's a centroid-based algorithm and the simplest unsupervised learning algorithm. This algorithm tries to minimize the variance of data points within a cluster. It's also how most people are introduced to unsupervised machine learning.

K-means is best used on smaller data sets because it iterates over all of the data points. That means it'll take more time to classify data points if there are a large amount of them in the data set. Since this is how k-means clusters data points, it doesn't scale well.

Reference: https://www.freecodecamp.org/news/8-clustering-algorithms-in-machine-learning-that-all-data-scientists-should-know/

### Dimension Reduction Techniques

### PCA
PCA is defined as an orthogonal linear transformation that transforms the data to a new coordinate system such that the greatest variance by some scalar projection of the data comes to lie on the first coordinate (called the first principal component), the second greatest variance on the second coordinate, and so on.

https://towardsdatascience.com/principal-component-analysis-pca-explained-visually-with-zero-math-1cbf392b9e7d#:~:text=PCA%20is%20defined%20as%20an,second%20coordinate%2C%20and%20so%20on.
## Stochastic models

### Markov Chains
<img src="https://user-images.githubusercontent.com/43540613/172156982-0dc7ae61-8155-4c03-97ca-8bea67113f0c.png" width="500"/>

The main goal of the Markov process is to identify the probability of transitioning from one state to another. This is one of the main appeals to Markov, that the future state of a stochastic variable is only dependent on its present state.
Markov Chains are exceptionally useful in order to model a discrete-time, discrete space Stochastic Process of various domains like Finance (stock price movement), NLP Algorithms (Finite State Transducers, Hidden Markov Model for POS Tagging), or even in Engineering Physics (Brownian motion). 

Great video going through the basics:
- https://www.youtube.com/watch?v=i3AkTO9HLXo

And a post with more information:
- https://towardsdatascience.com/introduction-to-markov-chains-50da3645a50d

## Other

### Bayesian Statistics
 The solution is a statistical technique called Bayesian inference. This technique begins with our stating prior beliefs about the system being modelled, allowing us to encode expert opinion and domain-specific knowledge into our system. These beliefs are combined with data to constrain the details of the model. Then, when used to make a prediction, the model doesn’t give one answer, but rather a distribution of likely answers, allowing us to assess risks.
https://towardsdatascience.com/what-is-bayesian-statistics-used-for-37b91c2c257c

## Data comparison
### ANOVA 
<img src="https://user-images.githubusercontent.com/43540613/173771587-b15ccd3c-b0c1-4192-988c-d5c5b68f6799.png" width="500"/>
Analysis of variance (ANOVA) is an analysis tool used in statistics that splits an observed aggregate variability found inside a data set into two parts: systematic factors and random factors. The systematic factors have a statistical influence on the given data set, while the random factors do not. Analysts use the ANOVA test to determine the influence that independent variables have on the dependent variable in a regression study.

There are two main types of ANOVA: one-way (or unidirectional) and two-way. There also variations of ANOVA. For example, MANOVA (multivariate ANOVA) differs from ANOVA as the former tests for multiple dependent variables simultaneously while the latter assesses only one dependent variable at a time. One-way or two-way refers to the number of independent variables in your analysis of variance test. A one-way ANOVA evaluates the impact of a sole factor on a sole response variable. It determines whether all the samples are the same. The one-way ANOVA is used to determine whether there are any statistically significant differences between the means of three or more independent (unrelated) groups.

A two-way ANOVA is an extension of the one-way ANOVA. With a one-way, you have one independent variable affecting a dependent variable. With a two-way ANOVA, there are two independents. For example, a two-way ANOVA allows a company to compare worker productivity based on two independent variables, such as salary and skill set. It is utilized to observe the interaction between the two factors and tests the effect of two factors at the same time.

Reference: https://www.investopedia.com/terms/a/anova.asp

## Ensemble Learning 
Ensemble learning algorithms combine the predictions of two or more models.

The idea of ensemble learning is closely related to the idea of the “wisdom of crowds“. This is where many different independent decisions, choices or estimates are combined into a final outcome that is often more accurate than any single contribution. They lead to better robustness and better predictions in our data.

We now have standardised models that apply ensemble learning for us. For example XGBoost, random forest, weighted averages.
https://machinelearningmastery.com/gentle-introduction-xgboost-applied-machine-learning/

### Bagging and Boosting
Bagging helps to decrease the model’s variance.
Boosting helps to decrease the model’s bias.

#### Bagging
<img src="https://user-images.githubusercontent.com/43540613/176615637-59d06af8-36a1-4fe8-8c52-1bd8008232a8.png" width="500"/>

Bagging is a way to decrease the variance in the prediction by generating additional data for training from dataset using combinations with repetitions to produce multi-sets of the original data. The idea behind bagging is combining the results of multiple models (for instance, all decision trees) to get a generalized result. Now, bootstrapping comes into picture. Bagging (or Bootstrap Aggregating) technique uses these subsets (bags) to get a fair idea of the distribution (complete set). The size of subsets created for bagging may be less than the original set.

Bagging works as follows:
1. Multiple subsets are created from the original dataset, selecting observations with replacement.
2. A base model (weak model) is created on each of these subsets.
3. The models run in parallel and are independent of each other.
4. The final predictions are determined by combining the predictions from all the models.


#### Boosting
<img src="https://user-images.githubusercontent.com/43540613/176615019-3e83fc48-db57-4bf5-b452-3d6d3f5fb5e7.png" width="500"/>

Boosting is an iterative technique which adjusts the weight of an observation based on the last classification. Boosting is a sequential process, where each subsequent model attempts to correct the errors of the previous model. The succeeding models are dependent on the previous model. In this technique, learners are learned sequentially with early learners fitting simple models to the data and then analyzing data for errors. In other words, we fit consecutive trees (random sample) and at every step, the goal is to solve for net error from the prior tree. When an input is misclassified by a hypothesis, its weight is increased so that next hypothesis is more likely to classify it correctly. By combining the whole set at the end converts weak learners into better performing model. This is because the individual models would not perform well on the entire dataset, but they work well for some part of the dataset. Thus, each model actually boosts the performance of the ensemble.

Boosting works as follows:
1. A subset is created from the original dataset.
2. Initially, all data points are given equal weights.
3. A base model is created on this subset.
4. This model is used to make predictions on the whole dataset.
5. Errors are calculated using the actual values and predicted values.
6. The observations which are incorrectly predicted, are given higher weights. (Here, the three misclassified blue-plus points will be given higher weights)
7. Another model is created and predictions are made on the dataset. (This model tries to correct the errors from the previous model)
8. Similarly, multiple models are created, each correcting the errors of the previous model.
9. The final model (strong learner) is the weighted mean of all the models (weak learners).

https://gaussian37.github.io/ml-concept-bagging/
https://www.youtube.com/watch?v=UeYG64Hm7Es
https://www.kaggle.com/code/prashant111/bagging-vs-boosting/notebook
https://medium.com/swlh/boosting-and-bagging-explained-with-examples-5353a36eb78d

#### Bagging vs Boosting
Similarities between Bagging and Boosting

##### Similarities between Bagging and Boosting are as follows:-
Both are ensemble methods to get N learners from 1 learner.
Both generate several training data sets by random sampling.
Both make the final decision by averaging the N learners (or taking the majority of them i.e Majority Voting).
Both are good at reducing variance and provide higher stability.

##### Differences between Bagging and Boosting 

Differences between Bagging and Boosting are as follows:-
Bagging is the simplest way of combining predictions that belong to the same type while Boosting is a way of combining predictions that belong to the different types.
Bagging aims to decrease variance, not bias while Boosting aims to decrease bias, not variance.
In Baggiing each model receives equal weight whereas in Boosting models are weighted according to their performance.
In Bagging each model is built independently whereas in Boosting new models are influenced by performance of previously built models.
In Bagging different training data subsets are randomly drawn with replacement from the entire training dataset. In Boosting every new subsets contains the elements that were misclassified by previous models.
Bagging tries to solve over-fitting problem while Boosting tries to reduce bias.
If the classifier is unstable (high variance), then we should apply Bagging. If the classifier is stable and simple (high bias) then we should apply Boosting.
Bagging is extended to Random forest model while Boosting is extended to Gradient boosting.

#### N-learners (sampling for Boosting/Bagging)
Bagging and Boosting get N learners by generating additional data in the training stage.
N new training data sets are produced by random sampling with replacement from the original set.
By sampling with replacement some observations may be repeated in each new training data set.
In the case of Bagging, any element has the same probability to appear in a new data set.
However, for Boosting the observations are weighted and therefore some of them will take part in the new sets more often.
These multiple sets are used to train the same learner algorithm and therefore different classifiers are produced.


## General References
https://www.linkedin.com/pulse/how-decide-which-model-use-anil-mahanty/

