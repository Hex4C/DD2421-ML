# Lab 3

Authors: Jesper Lindeberg, Linus Markström

Date: Today :D

## Assignment 1 
Guassian plot for the ML parameter calculations, visualising the mean and variance of the classes.

![Guassian plot](images/1_Plot_Guassian.png)

## Assigment 2
Nothing to say really :P 🤩🤩

## Assignment 3
The Iris dataset performs really well, while the Vowel dataset performs less impresive.

| Dataset   | Accuracy |
| --------- | ---------|
| Iris      | 89 %     |
| Vowel     | 64.7 %   |

The images below display the estimated decision boundry from the trained bayesian classifiers. The poor performance of the Vowel dataset could be due to the high amount of classes.
 
![Iris decision boundry](images/3_Descision_boundary_iris.png)

### 1)

For example in cases of high-dimension datasets (more features than samples), for catogorical features or text classification (treat as independat, when actually not).

### 2) 

We could optimize the classifier by using the non-naive bayes classifier meaning we don't assume independence between the classes and account for possible dependence between them. 

However, a non-naive bayesian classifier, General Bayesian Classifier, is more computationally heavy requires more data to see possible dependence between classes. 


## Assignment 4