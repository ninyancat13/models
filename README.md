# Which model to use when?

<img src="images/screenshot.png" width="500"/>
Note: something like random forest or decision trees can be used for both continuous and categorical problems

## Supervised Models
### Decision trees
<img src="https://user-images.githubusercontent.com/43540613/172396826-af4abd77-261e-45f7-9855-9db13e53db48.png" width="350"/>

Decision Tree algorithm belongs to the family of supervised learning algorithms. Unlike other supervised learning algorithms, the decision tree algorithm can be used for solving regression and classification problems too. The goal of using a Decision Tree is to create a training model that can use to predict the class or value of the target variable by learning simple decision rules inferred from prior data(training data).

### Random Forest
<img src="https://user-images.githubusercontent.com/43540613/172595075-cc931c80-7af9-4bfe-8adf-abcdf7305a79.png" width="500"/>

The random forest algorithm is an extension of the bagging method as it utilizes both bagging and feature randomness to create an uncorrelated forest of decision trees. Feature randomness, also known as feature bagging or "the random subspace method", generates a random subset of features, which ensures low correlation among decision trees. This is a key difference between decision trees and random forests. While decision trees consider all the possible feature splits, random forests only select a subset of those features. By accounting for all the potential variability in the data, we can reduce the risk of overfitting, bias, and overall variance, resulting in more precise predictions.

Note: What is an ensemble method?
Ensemble learning methods are made up of a set of classifiers - e.g. decision trees and their predictions are aggregated to identify the most popular result. The most well-known ensemble methods are bagging, also known as bootstrap aggregation, and boosting. In bagging, a random sample of data in a training set is selected with replacement - meaning that the individual data points can be chosen more than once. After several data samples are generated, these models are then trained independently, and depending on the type of task - i.e. regression or classification - the average or majority of those predictions yield a more accurate estimate. This approach is commonly used to reduce variance within a noisy dataset.

Reference: https://www.ibm.com/cloud/learn/random-forest

## Unsupervised Models


## Stochastic models
### Markov Chains
<img src="https://user-images.githubusercontent.com/43540613/172156982-0dc7ae61-8155-4c03-97ca-8bea67113f0c.png" width="500"/>

The main goal of the Markov process is to identify the probability of transitioning from one state to another. This is one of the main appeals to Markov, that the future state of a stochastic variable is only dependent on its present state.
Markov Chains are exceptionally useful in order to model a discrete-time, discrete space Stochastic Process of various domains like Finance (stock price movement), NLP Algorithms (Finite State Transducers, Hidden Markov Model for POS Tagging), or even in Engineering Physics (Brownian motion). 

- https://www.youtube.com/watch?v=i3AkTO9HLXo
- https://towardsdatascience.com/introduction-to-markov-chains-50da3645a50d

## References
https://www.linkedin.com/pulse/how-decide-which-model-use-anil-mahanty/

