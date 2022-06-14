# Which model to use when?

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

### Logistic Regression
<img src="https://user-images.githubusercontent.com/43540613/172834151-b83706d1-713a-4153-819b-fdbd098dac5e.png" width="500"/>

Logistic regression is a process of modeling the probability of a discrete outcome given an input variable. The most common logistic regression models a binary outcome; something that can take two values such as true/false, yes/no, and so on.

#### Should I use logistic regression or a decision tree?
Decision Trees are non-linear classifiers; they do not require data to be linearly separable. When you are sure that your data set divides into two separable parts, then use a Logistic Regression. If you're not sure, then go with a Decision Tree. A Decision Tree will take care of both.

### Linear Regression
<img src="https://user-images.githubusercontent.com/43540613/173571597-06618223-084e-464e-8bdd-9d9b1502a125.png" width="500"/>

Linear regression attempts to model the relationship between two variables by fitting a linear equation to observed data. One variable is considered to be an explanatory variable, and the other is considered to be a dependent variable. For example, a modeler might want to relate the weights of individuals to their heights using a linear regression model. Before attempting to fit a linear model to observed data, a modeler should first determine whether or not there is a relationship between the variables of interest. This does not necessarily imply that one variable causes the other (for example, higher SAT scores do not cause higher college grades), but that there is some significant association between the two variables. A scatterplot can be a helpful tool in determining the strength of the relationship between two variables. If there appears to be no association between the proposed explanatory and dependent variables (i.e., the scatterplot does not indicate any increasing or decreasing trends), then fitting a linear regression model to the data probably will not provide a useful model. A valuable numerical measure of association between two variables is the correlation coefficient, which is a value between -1 and 1 indicating the strength of the association of the observed data for the two variables.

A linear regression line has an equation of the form Y = a + bX, where X is the explanatory variable and Y is the dependent variable. The slope of the line is b, and a is the intercept (the value of y when x = 0).


## Unsupervised Models


## Stochastic models

### Markov Chains
<img src="https://user-images.githubusercontent.com/43540613/172156982-0dc7ae61-8155-4c03-97ca-8bea67113f0c.png" width="500"/>

The main goal of the Markov process is to identify the probability of transitioning from one state to another. This is one of the main appeals to Markov, that the future state of a stochastic variable is only dependent on its present state.
Markov Chains are exceptionally useful in order to model a discrete-time, discrete space Stochastic Process of various domains like Finance (stock price movement), NLP Algorithms (Finite State Transducers, Hidden Markov Model for POS Tagging), or even in Engineering Physics (Brownian motion). 

Great video going through the basics:
- https://www.youtube.com/watch?v=i3AkTO9HLXo

And a post with more information:
- https://towardsdatascience.com/introduction-to-markov-chains-50da3645a50d

## General References
https://www.linkedin.com/pulse/how-decide-which-model-use-anil-mahanty/

