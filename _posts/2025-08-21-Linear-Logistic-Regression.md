---
layout: post
title: A complete understanding of Linear and Logistic Regression
date: 2025-08-21 12:50:50
description: In depth look at Linear & Logistic Regression
tags: formatting links
categories: sample-posts
---

# Linear Regression Model

What assumption do we have in Linear Regression?

The assumption that there is a relationship between Y (target variable) and X (features)
Or simply: 
$$\hat{Y} = W_1X_1 + W_2X_2 + W_3X_3 + \dots + W_nX_n$$
$$\hat{Y} = W^{T}X + W_0$$

Linear Regression predicts a continuous numeric value (e.g., predicting house prices, temperature, weight).

# Logsitic Regression Model

What assumption do we have in Logistic Regression?

Logistic Regression assumes a linear relationship between features and the log-odds (logit) of the outcome.
 
$$W^{T}X $$
$$P[Y=1 | X=x] = \sigma(W^{T}X) >> \frac{1}{1+e^{-W_1X_1 + W_2X_2 + W_3X_3+ \dots+ W_nX_n}}$$

Logistic Regression predicts a probability that maps to a categorical outcome (most often binary classification, like "spam vs not spam").

# The difference in short
Linear Regression predicts continuous values with a straight line fit.

Logistic Regression predicts probabilities for categorical outcomes using a sigmoid/logistic curve.

# WIP