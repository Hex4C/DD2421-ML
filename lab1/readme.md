# Lab 1

Lab 1 decision trees

Author: Jesper Lindeberg, Linus Markström

Date: 27/1 - 2026

## Assignment 0

The hardest set to classify is MONK-2 since it can't be efficiently mapped by a decision tree whilst MONK-1 and MONK-3 can be. We need 2 element to exclusively be equal to 1 which is essentially the XOR problem, which is aknow hard problem for descision trees [scikit](https://scikit-learn.org/stable/modules/tree.html).

## Assignment 1

| Monk   | Entropy            |
| ------ | ------------------ |
| MONK 1 | 1                  |
| MONK 2 | 0.957117428264771  |
| MONK 3 | 0.9998061328047111 |

## Assignment 2

high entropy = low predictability
low entropy = high predictability

In uniform distribution, all cases has an equal probability, this makes it hard to predict the outcome. This will result in a low predictability and therefore a high entropy. In non-uniform distributions, some cases will have higher chance of happening, which result in a possibility to predict these higher probable outcomes. Therefore, a non-uniform distribution is more predicable and will result in low entropy.

An example for a high entropy would be a fair die. Whilst an example of a distribution with low entropy would be a normal gaussian distribution.

## Assignment 3

| Dataset | a1     | a2     | a3     | a4     | a5     | a6     |
| ------- | ------ | ------ | ------ | ------ | ------ | ------ |
| Monk-1  | 0.0753 | 0.0058 | 0.0047 | 0.0263 | 0.287  | 0.0008 |
| Monk-2  | 0.0038 | 0.0025 | 0.0011 | 0.0157 | 0.0173 | 0.0062 |
| Monk-3  | 0.0071 | 0.2937 | 0.0008 | 0.0029 | 0.2559 | 0.0071 |

Monk 1: Max attribute 4 gain: 0.287

Monk 2: Max attribute 4 gain: 0.0173

Monk 3: Max attribute 1 gain: 0.2937

For Monk 1 we choose attribute 4, for Monk 2 attribute 4 and for Monk 3 attribute 1. This based on the information gain, which these attributes maximizes.

## Assignement 4

