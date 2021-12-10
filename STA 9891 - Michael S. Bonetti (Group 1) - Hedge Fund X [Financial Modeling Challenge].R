###############################################################################
# STA 9891 - Machine Learning for Data Mining
# CUNY Bernard M. Baruch College
# Prof. Kamiar Rahnama Rad
# Final Project

# Hedge Fund X: Financial Modeling Challenge
# Group 1 - Michael S. Bonetti
# December 6, 2021
###############################################################################



# Clear Log, Environment, Plots and Memory
rm(list = ls())
cat("\014")
graphics.off()
gc()



###############################################################################
# I. Load Libraries
###############################################################################



library(class)
library(coefplot)
library(dplyr)
library(e1071)
library(ggplot2)
library(glmnet)
library(grid)
library(gridExtra)
library(ISLR)
library(latex2exp)
library(MASS)
library(pROC)
library(randomForest)
library(RColorBrewer)
library(ROCR)
library(rmutil)
library(tictoc)
library(tidyverse)



###############################################################################
# II. Loading & Cleaning Data
###############################################################################



# Running locally and reading file (n = 10000, p = 88) [Full dataset]
data10000    =    read.csv("C:\\Users\\Blwndrpwrmlk\\Dropbox\\Baruch\\deepanalytics_dataset.csv",header = TRUE)
y            =    factor(data10000$target)

# Testing dataset for imbalance
sum(y == 0)
sum(y == 1)

# Using subset of full dataset (n = 1000, p = 88), due to time complexity of SVM is n2.

# Run 1 Dataset - Static 1,000 observations
#data1000     =    read.csv("C:\\Users\\Blwndrpwrmlk\\Dropbox\\Baruch\\deepanalytics_dataset_1000.csv",header = TRUE)

# Run 2 Dataset - Randomly-chosen 1,000 observations
data1000     =    data10000[, -c(1:2)]                                           # Remove first two (non-predictive) attributes
data1000     =    data1000[sample(nrow(data1000), 1000),]                        # Subset for 1,000 observations

X            =    model.matrix(target ~ ., data1000)[, -1]
y            =    factor(data1000$target)

# Testing subset dataset for imbalance
sum(y == 0)
sum(y == 1)

# Data Preparation
n             =    dim(X)[1]                                                     # Sample Size
p             =    dim(X)[2]                                                     # Number of predictors/features
S             =    50                                                            # Iterations

learn.pct1    =    0.5                                                           # 0.5n
learn.pct2    =    0.9                                                           # 0.9n
learn.pct     =    c(learn.pct1, learn.pct2)
random_order  =    matrix(0, nrow = S, ncol = n)                                 # Random order matrix

set.seed(1) 

X             =    scale(X)

# Setting up error matrices for RF, SVM, Logistic, LASSO, Ridge, & Elnet
Err.rf        =    matrix(0, nrow = S, ncol = 14) 
Err.svm       =    matrix(0, nrow = S, ncol = 14) 
Err.logistic  =    matrix(0, nrow = S, ncol = 14) 
Err.LASSO     =    matrix(0, nrow = S, ncol = 14) 
Err.ridge     =    matrix(0, nrow = S, ncol = 14)
Err.elnet     =    matrix(0, nrow = S, ncol = 14)

# Setting up AUC matrices for RF, SVM, Logistic, LASSO, Ridge and Elnet
AUC.rf        =    matrix(0, nrow = S, ncol = 4)
AUC.svm       =    matrix(0, nrow = S, ncol = 4)
AUC.logistic  =    matrix(0, nrow = S, ncol = 4)
AUC.LASSO     =    matrix(0, nrow = S, ncol = 4)
AUC.ridge     =    matrix(0, nrow = S, ncol = 4)
AUC.elnet     =    matrix(0, nrow = S, ncol = 4)



###############################################################################
# III. Creating ML Models 
###############################################################################



#################################################
# Q2(a)-(c). Fit models 50 times with 6 methods

##################
# START 50x LOOP
##################

for (s in 1:S) {
  
  random_order[s,]  =  sample(n)                                                 # Q2a. Randomly split data into testing and training sets
  
  for (i in learn.pct) {
    
    ptm          =  proc.time()                                                  # Record time
    
    n.train      =  floor(n*i)
    n.test       =  n - n.train
    trainSet     =  random_order[s,][1:n.train]
    testSet      =  random_order[s,][(1 + n.train):n] 
    X.train      =  X[trainSet, ]
    y.train      =  y[trainSet]
    X.test       =  X[testSet, ]
    y.test       =  y[testSet]
    y.os.train   =  y.train                                                      # Initialize the over-sampled set to train models
    X.os.train   =  X.train                                                      # Initialize the over-sampled set to train models
    
    #######################################################
    # To account for imbalance, data will be over-sampled
    # (w/ replacement) to become balanced
    imbalance     =     FALSE   
    if (imbalance == TRUE) {
      index.yis0      =      which(y.train == 0)                                 # Identify index of points with label 0
      index.yis1      =      which(y.train == 1)                                 # Identify index of points with label 1
      n.train.1       =      length(index.yis1)
      n.train.0       =      length(index.yis0)
      if (n.train.1 > n.train.0) {                                               # Needing more 0s in training set (Over-sampling with replacement)
        more.train    =      sample(index.yis0, size = n.train.1-n.train.0,
                                    replace = TRUE)
      }         else {                                                           # Needing more 1s in training set (Over-sampling with replacement)
        more.train    =      sample(index.yis1, size = n.train.0-n.train.1,
                                    replace = TRUE)
      }
      #############################################################
      # The code below CORRECTLY over-samples the training set and
      # stores into y.train and X.train
      y.os.train      =      as.factor(c(y.train, y.train[more.train])-1) 
      X.os.train      =      rbind2(X.train, X.train[more.train,])    
      
    }
    
    os.train.data             =   data.frame(X.os.train, as.factor(y.os.train))
    train.data                =   data.frame(X.train, as.factor(y.train))
    test.data                 =   data.frame(X.test, as.factor(y.test))
    names(os.train.data)[89]  =   "target"
    names(train.data)[89]     =   "target"
    names(test.data)[89]      =   "target"
    
    #####################
    # Random Forest (RF)
    #####################
    rf.fit         =   randomForest(target~., data = os.train.data,
                                    mtry = sqrt(p), importance = TRUE)
    y.train.hat    =   predict(rf.fit, newdata = train.data)
    y.test.hat     =   predict(rf.fit, newdata = test.data)
    
    if (i == learn.pct1) {
      Err.rf[s,1]  =   mean(y.train != y.train.hat)
      Err.rf[s,3]  =   mean(y.test != y.test.hat)
      
      Err.rf[s,7]  =   mean(1 == y.train.hat[y.train == 0])                      # False Positive
      Err.rf[s,8]  =   mean(0 == y.train.hat[y.train == 1])                      # False Negative
      Err.rf[s,9]  =   mean(1 == y.test.hat[y.test == 0])                        # False Positive
      Err.rf[s,10] =   mean(0 == y.test.hat[y.test == 1])                        # False Negative
      
      AUC.rf[s,1]  =   auc(train.data$target, as.numeric(y.train.hat))           # RF AUC Train
      AUC.rf[s,2]  =   auc(test.data$target, as.numeric(y.test.hat))             # RF AUC Test
      
    } else {
      Err.rf[s,2]  =   mean(y.train != y.train.hat)
      Err.rf[s,4]  =   mean(y.test != y.test.hat)
      
      Err.rf[s,11] =   mean(1 == y.train.hat[y.train == 0])                      # False Positive
      Err.rf[s,12] =   mean(0 == y.train.hat[y.train == 1])                      # False Negative
      Err.rf[s,13] =   mean(1 == y.test.hat[y.test == 0])                        # False Positive
      Err.rf[s,14] =   mean(0 == y.test.hat[y.test == 1])                        # False Negative
      
      AUC.rf[s,3]  =   auc(train.data$target, as.numeric(y.train.hat))           # RF AUC Train
      AUC.rf[s,4]  =   auc(test.data$target, as.numeric(y.test.hat))             # RF AUC Test
      
    }

    #############
    # Radial SVM
    #############
    tune.svm    =   tune(svm, target ~ ., data = os.train.data,
                         kernel = "radial",
                         ranges = list(cost = 10^seq(-2,2,length.out = 5),
                                       gamma = 10^seq(-2,2,length.out = 5)))
    SVM.fit     =   tune.svm$best.model
    y.train.hat =   predict(SVM.fit, newdata = train.data)
    y.test.hat  =   predict(SVM.fit, newdata = test.data)
    
    if (i == learn.pct1) {
      Err.svm[s,1]    =     mean(y.train != y.train.hat)
      Err.svm[s,3]    =     mean(y.test != y.test.hat)
      Err.svm[s,5]    =     tune.svm$best.performance
      
      Err.svm[s,7]    =     mean(1 == y.train.hat[y.train == 0])                 # False Positive
      Err.svm[s,8]    =     mean(0 == y.train.hat[y.train == 1])                 # False Negative
      Err.svm[s,9]    =     mean(1 == y.test.hat[y.test == 0])                   # False Positive
      Err.svm[s,10]   =     mean(0 == y.test.hat[y.test == 1])                   # False Negative
      
      AUC.svm[s,1]    =     auc(train.data$target, as.numeric(y.train.hat))      # SVM AUC Train
      AUC.svm[s,2]    =     auc(test.data$target, as.numeric(y.test.hat))        # SVM AUC Test
      
    } else {
      Err.svm[s,2]    =     mean(y.train != y.train.hat)
      Err.svm[s,4]    =     mean(y.test != y.test.hat)
      Err.svm[s,6]    =     tune.svm$best.performance
      
      Err.svm[s,11]   =     mean(1 == y.train.hat[y.train == 0])                 # False Positive
      Err.svm[s,12]   =     mean(0 == y.train.hat[y.train == 1])                 # False Negative
      Err.svm[s,13]   =     mean(1 == y.test.hat[y.test == 0])                   # False Positive
      Err.svm[s,14]   =     mean(0 == y.test.hat[y.test == 1])                   # False Negative
      
      AUC.svm[s,3]    =     auc(train.data$target, as.numeric(y.train.hat))      # SVM AUC Train
      AUC.svm[s,4]    =     auc(test.data$target, as.numeric(y.test.hat))        # SVM AUC Test
      
    }
    
    ######################
    # Logistic Regression
    ######################
    logistic.fit          =   glm(target ~ ., os.train.data,
                                  family = "binomial")
    logistic.probs.train  =   predict(logistic.fit, train.data, "response")
    logistic.probs.test   =   predict(logistic.fit, test.data, "response")
    y.train.hat           =   rep(0, length(logistic.probs.train))
    y.test.hat            =   rep(0, length(logistic.probs.test))
    y.train.hat[logistic.probs.train > 0.5] = 1
    y.test.hat[logistic.probs.test > 0.5]   = 1
    
    if (i == learn.pct1) {
      Err.logistic[s,1]    =     mean(y.train != y.train.hat)
      Err.logistic[s,3]    =     mean(y.test != y.test.hat)
      
      Err.logistic[s,7]    =     mean(1 == y.train.hat[y.train == 0])            # False Positive
      Err.logistic[s,8]    =     mean(0 == y.train.hat[y.train == 1])            # False Negative
      Err.logistic[s,9]    =     mean(1 == y.test.hat[y.test == 0])              # False Positive
      Err.logistic[s,10]   =     mean(0 == y.test.hat[y.test == 1])              # False Negative
      
      AUC.logistic[s,1]    =     auc(train.data$target, as.numeric(y.train.hat)) # Logistic AUC Train
      AUC.logistic[s,2]    =     auc(test.data$target, as.numeric(y.test.hat))   # Logistic AUC Test
      
    } else {
      Err.logistic[s,2]    =     mean(y.train != y.train.hat)
      Err.logistic[s,4]    =     mean(y.test != y.test.hat)
      
      Err.logistic[s,11]   =     mean(1 == y.train.hat[y.train == 0])            # False Positive
      Err.logistic[s,12]   =     mean(0 == y.train.hat[y.train == 1])            # False Negative
      Err.logistic[s,13]   =     mean(1 == y.test.hat[y.test == 0])              # False Positive
      Err.logistic[s,14]   =     mean(0 == y.test.hat[y.test == 1])              # False Negative
      
      AUC.logistic[s,3]    =     auc(train.data$target, as.numeric(y.train.hat)) # Logistic AUC Train
      AUC.logistic[s,4]    =     auc(test.data$target, as.numeric(y.test.hat))   # Logistic AUC Test
      
    }
    
    ##############################################################
    # LASSO Logistic regression, optimized using cross-validation
    ##############################################################
    #m                 =     25
    m                 =     50
    LASSO.CV          =     cv.glmnet(X.os.train, y.os.train,
                                      family = "binomial", alpha = 1,
                                      nfolds = 10, type.measure = "class")
    LASSO.fit         =     glmnet(X.os.train, y.os.train,
                                   lambda = LASSO.CV$lambda.min,
                                   family = "binomial", alpha = 1)
    y.train.hat       =     predict(LASSO.fit, newx = X.train, type = "class")
    y.test.hat        =     predict(LASSO.fit, newx = X.test, type = "class")

    if (i == learn.pct1) {
      Err.LASSO[s,1]    =     mean(y.train != y.train.hat)
      Err.LASSO[s,3]    =     mean(y.test != y.test.hat)
      Err.LASSO[s,5]    =     min(LASSO.CV$cvm)
      
      Err.LASSO[s,7]    =     mean(1 == y.train.hat[y.train == 0])               # False Positive
      Err.LASSO[s,8]    =     mean(0 == y.train.hat[y.train == 1])               # False Negative
      Err.LASSO[s,9]    =     mean(1 == y.test.hat[y.test == 0])                 # False Positive
      Err.LASSO[s,10]   =     mean(0 == y.test.hat[y.test == 1])                 # False Negative
      
      AUC.LASSO[s,1]    =     auc(train.data$target, as.numeric(y.train.hat))    # LASSO AUC Train
      AUC.LASSO[s,2]    =     auc(test.data$target, as.numeric(y.test.hat))      # LASSO AUC Test
      
    } else {
      Err.LASSO[s,2]    =     mean(y.train != y.train.hat)
      Err.LASSO[s,4]    =     mean(y.test != y.test.hat)
      Err.LASSO[s,6]    =     min(LASSO.CV$cvm)
      
      Err.LASSO[s,11]   =     mean(1 == y.train.hat[y.train == 0])               # False Positive
      Err.LASSO[s,12]   =     mean(0 == y.train.hat[y.train == 1])               # False Negative
      Err.LASSO[s,13]   =     mean(1 == y.test.hat[y.test == 0])                 # False Positive
      Err.LASSO[s,14]   =     mean(0 == y.test.hat[y.test == 1])                 # False Negative
      
      AUC.LASSO[s,3]    =     auc(train.data$target, as.numeric(y.train.hat))    # LASSO AUC Train
      AUC.LASSO[s,4]    =     auc(test.data$target, as.numeric(y.test.hat))      # LASSO AUC Test
 
    }
    
    ##############################################################
    # Ridge Logistic regression, optimized using cross-validation
    ##############################################################
    #m                 =     25
    m                 =     50
    ridge.CV          =     cv.glmnet(X.os.train, y.os.train,
                                      family = "binomial", alpha = 0,
                                      nfolds = 10, type.measure = "class")
    ridge.fit         =     glmnet(X.os.train, y.os.train,
                                   lambda = ridge.CV$lambda.min,
                                   family = "binomial", alpha = 0)
    y.train.hat       =     predict(ridge.fit, newx = X.train, type = "class")
    y.test.hat        =     predict(ridge.fit, newx = X.test, type = "class")

    if (i == learn.pct1) {
      Err.ridge[s,1]    =     mean(y.train != y.train.hat)
      Err.ridge[s,3]    =     mean(y.test != y.test.hat)
      Err.ridge[s,5]    =     min(ridge.CV$cvm)
      
      Err.ridge[s,7]    =     mean(1 == y.train.hat[y.train == 0])               # False Positive
      Err.ridge[s,8]    =     mean(0 == y.train.hat[y.train == 1])               # False Negative
      Err.ridge[s,9]    =     mean(1 == y.test.hat[y.test == 0])                 # False Positive
      Err.ridge[s,10]   =     mean(0 == y.test.hat[y.test == 1])                 # False Negative
      
      AUC.ridge[s,1]    =     auc(train.data$target, as.numeric(y.train.hat))    # Ridge AUC Train
      AUC.ridge[s,2]    =     auc(test.data$target, as.numeric(y.test.hat))      # Ridge AUC Test
      
    } else {
      Err.ridge[s,2]    =     mean(y.train != y.train.hat)
      Err.ridge[s,4]    =     mean(y.test != y.test.hat)
      Err.ridge[s,6]    =     min(ridge.CV$cvm)
      
      Err.ridge[s,11]   =     mean(1 == y.train.hat[y.train == 0])               # False Positive
      Err.ridge[s,12]   =     mean(0 == y.train.hat[y.train == 1])               # False Negative
      Err.ridge[s,13]   =     mean(1 == y.test.hat[y.test == 0])                 # False Positive
      Err.ridge[s,14]   =     mean(0 == y.test.hat[y.test == 1])                 # False Negative
      
      AUC.ridge[s,3]    =     auc(train.data$target, as.numeric(y.train.hat))    # Ridge AUC Train
      AUC.ridge[s,4]    =     auc(test.data$target, as.numeric(y.test.hat))      # Ridge AUC Test
      
    } 
    
    #########################################################################
    # Elastic-net (EN) Logistic regression, optimized using cross-validation
    #########################################################################
    #m                 =     25
    m                 =     50
    elnet.CV          =     cv.glmnet(X.os.train, y.os.train,
                                      family = "binomial", alpha = 0.5,
                                      nfolds = 10, type.measure = "class")
    elnet.fit         =     glmnet(X.os.train, y.os.train,
                                   lambda = elnet.CV$lambda.min,
                                   family = "binomial", alpha = 0.5)
    y.train.hat       =     predict(elnet.fit, newx = X.train, type = "class")
    y.test.hat        =     predict(elnet.fit, newx = X.test, type = "class")
   
    if (i == learn.pct1) {
      Err.elnet[s,1]    =     mean(y.train != y.train.hat)
      Err.elnet[s,3]    =     mean(y.test != y.test.hat)
      Err.elnet[s,5]    =     min(elnet.CV$cvm)

      Err.elnet[s,7]    =     mean(1 == y.train.hat[y.train == 0])               # False Positive
      Err.elnet[s,8]    =     mean(0 == y.train.hat[y.train == 1])               # False Negative
      Err.elnet[s,9]    =     mean(1 == y.test.hat[y.test == 0])                 # False Positive
      Err.elnet[s,10]   =     mean(0 == y.test.hat[y.test == 1])                 # False Negative
      
      AUC.elnet[s,1]    =     auc(train.data$target, as.numeric(y.train.hat))    # Elnet AUC Train
      AUC.elnet[s,2]    =     auc(test.data$target, as.numeric(y.test.hat))      # Elnet AUC Test

    } else {
      Err.elnet[s,2]    =     mean(y.train != y.train.hat)
      Err.elnet[s,4]    =     mean(y.test != y.test.hat)
      Err.elnet[s,6]    =     min(elnet.CV$cvm)

      Err.elnet[s,11]   =     mean(1 == y.train.hat[y.train == 0])               # False Positive
      Err.elnet[s,12]   =     mean(0 == y.train.hat[y.train == 1])               # False Negative
      Err.elnet[s,13]   =     mean(1 == y.test.hat[y.test == 0])                 # False Positive
      Err.elnet[s,14]   =     mean(0 == y.test.hat[y.test == 1])                 # False Negative
      
      AUC.elnet[s,3]    =     auc(train.data$target, as.numeric(y.train.hat))    # Elnet AUC Train
      AUC.elnet[s,4]    =     auc(test.data$target, as.numeric(y.test.hat))      # Elnet AUC Test
      
    }
    
    ptm       =     proc.time() - ptm                                            # Output time
    if (i == learn.pct1) {
      time1   =     ptm["elapsed"]
    } else {
      time2   =     ptm["elapsed"]
    } 
    
  }
  
  cat(sprintf("s=%1.f: 
              %.1fn | time: %0.3f(sec) |
              Train:     RF = %.2f, SVM = %.2f, Logistic = %.2f, LASSO = %.2f, Ridge = %.2f, Elnet = %.2f
              Test:      RF = %.2f, SVM = %.2f, Logistic = %.2f, LASSO = %.2f, Ridge = %.2f, Elnet = %.2f
              Min CV:    RF = %.2f, SVM = %.2f, Logistic = %.2f, LASSO = %.2f, Ridge = %.2f, Elnet = %.2f
              AUC Train: RF = %.2f, SVM = %.2f, Logistic = %.2f, LASSO = %.2f, Ridge = %.2f, Elnet = %.2f
              AUC Test:  RF = %.2f, SVM = %.2f, Logistic = %.2f, LASSO = %.2f, Ridge = %.2f, Elnet = %.2f
              %.1fn | time: %0.3f(sec) |
              Train:     RF = %.2f, SVM = %.2f, Logistic = %.2f, LASSO = %.2f, Ridge = %.2f, Elnet = %.2f
              Test:      RF = %.2f, SVM = %.2f, Logistic = %.2f, LASSO = %.2f, Ridge = %.2f, Elnet = %.2f
              Min CV:    RF = %.2f, SVM = %.2f, Logistic = %.2f, LASSO = %.2f, Ridge = %.2f, Elnet = %.2f
              AUC Train: RF = %.2f, SVM = %.2f, Logistic = %.2f, LASSO = %.2f, Ridge = %.2f, Elnet = %.2f
              AUC Test:  RF = %.2f, SVM = %.2f, Logistic = %.2f, LASSO = %.2f, Ridge = %.2f, Elnet = %.2f\n",s,
              learn.pct1,time1,
              Err.rf[s,1],Err.svm[s,1],Err.logistic[s,1],Err.LASSO[s,1],Err.ridge[s,1],Err.elnet[s,1],
              Err.rf[s,3],Err.svm[s,3],Err.logistic[s,3],Err.LASSO[s,3],Err.ridge[s,3],Err.elnet[s,3],
              Err.rf[s,5],Err.svm[s,5],Err.logistic[s,5],Err.LASSO[s,5],Err.ridge[s,5],Err.elnet[s,5],
              AUC.rf[s,1],AUC.svm[s,1],AUC.logistic[s,1],AUC.LASSO[s,1],AUC.ridge[s,1],AUC.elnet[s,1],
              AUC.rf[s,2],AUC.svm[s,2],AUC.logistic[s,2],AUC.LASSO[s,2],AUC.ridge[s,2],AUC.elnet[s,2],
              learn.pct2,time2,
              Err.rf[s,2],Err.svm[s,2],Err.logistic[s,2],Err.LASSO[s,2],Err.ridge[s,2],Err.elnet[s,2],
              Err.rf[s,4],Err.svm[s,4],Err.logistic[s,4],Err.LASSO[s,4],Err.ridge[s,4],Err.elnet[s,4],
              Err.rf[s,6],Err.svm[s,6],Err.logistic[s,6],Err.LASSO[s,6],Err.ridge[s,6],Err.elnet[s,6],
              AUC.rf[s,3],AUC.svm[s,3],AUC.logistic[s,3],AUC.LASSO[s,3],AUC.ridge[s,3],AUC.elnet[s,3],
              AUC.rf[s,4],AUC.svm[s,4],AUC.logistic[s,4],AUC.LASSO[s,4],AUC.ridge[s,4],AUC.elnet[s,4]))
  
}

################
# END 50x LOOP
################

# Checkpoint 1
save.image(file = 'Checkpoint 1 - 50x Loop.RData')

# Saving sample order, error files and AUCs
write.csv(data.frame(random_order), "random_order.csv", row.names = FALSE)
write.csv(data.frame(Err.rf), "Err.rf.csv", row.names = FALSE)
write.csv(data.frame(Err.svm), "Err.svm.csv", row.names = FALSE)
write.csv(data.frame(Err.logistic), "Err.logistic.csv", row.names = FALSE)
write.csv(data.frame(Err.LASSO), "Err.LASSO.csv", row.names = FALSE)
write.csv(data.frame(Err.ridge), "Err.ridge.csv", row.names = FALSE)
write.csv(data.frame(Err.elnet), "Err.elnet.csv", row.names = FALSE)

# Saving AUCs
write.csv(data.frame(AUC.rf), "AUC.rf.csv", row.names = FALSE)
write.csv(data.frame(AUC.svm), "AUC.svm.csv", row.names = FALSE)
write.csv(data.frame(AUC.logistic), "AUC.logistic.csv", row.names = FALSE)
write.csv(data.frame(AUC.LASSO), "AUC.LASSO.csv", row.names = FALSE)
write.csv(data.frame(AUC.ridge), "AUC.ridge.csv", row.names = FALSE)
write.csv(data.frame(AUC.elnet), "AUC.elnet.csv", row.names = FALSE)



###############################################################################
# IV. Error Sets & AUCs
###############################################################################



# (Hide)
#####################
# # 0.5n Training Size
# err.train.pct1 = data.frame(c(rep("RF",S),rep("Radial SVM",S),rep("Logistic",S),
#                               rep("Logistic LASSO",S),rep("Logistic Ridge",S),rep("Logistic Elnet",S)), 
#                             c(Err.rf[,1],Err.svm[,1],Err.logistic[,1],
#                               Err.LASSO[,1],Err.ridge[,1],Err.elnet[,1]))
# 
# err.test.pct1  = data.frame(c(rep("RF",S),rep("Radial SVM",S),rep("Logistic",S),
#                               rep("Logistic LASSO",S),rep("Logistic Ridge",S),rep("Logistic Elnet",S)), 
#                             c(Err.rf[,3],Err.svm[,3],Err.logistic[,3],
#                               Err.LASSO[,3],Err.ridge[,3],Err.elnet[,3]))
# 
# err.minCV.pct1 = data.frame(c(rep("RF",S),rep("Radial SVM",S),rep("Logistic",S),
#                               rep("Logistic LASSO",S),rep("Logistic Ridge",S),rep("Logistic Elnet",S)), 
#                             c(Err.rf[,5],Err.svm[,5],Err.logistic[,5],
#                               Err.LASSO[,5],Err.ridge[,5],Err.elnet[,5]))
# 
# # Train False Positive (FP) - 0.5n
# err.train.fp.pct1 = data.frame(c(rep("RF",S),rep("Radial SVM",S),rep("Logistic",S),
#                                  rep("Logistic LASSO",S),rep("Logistic Ridge",S),rep("Logistic Elnet",S)), 
#                                c(Err.rf[,7],Err.svm[,7],Err.logistic[,7],
#                                  Err.LASSO[,7],Err.ridge[,7],Err.elnet[,7]))
# 
# # Train False Negative (FN) - 0.5n
# err.train.fn.pct1 = data.frame(c(rep("RF",S),rep("Radial SVM",S),rep("Logistic",S),
#                                  rep("Logistic LASSO",S),rep("Logistic Ridge",S),rep("Logistic Elnet",S)), 
#                                c(Err.rf[,8],Err.svm[,8],Err.logistic[,8],
#                                  Err.LASSO[,8],Err.ridge[,8],Err.elnet[,8]))
# 
# # Test False Positive (FP) - 0.5n
# err.test.fp.pct1 = data.frame(c(rep("RF",S),rep("Radial SVM",S),rep("Logistic",S),
#                                 rep("Logistic LASSO",S),rep("Logistic Ridge",S),rep("Logistic Elnet",S)), 
#                               c(Err.rf[,9],Err.svm[,9],Err.logistic[,9],
#                                 Err.LASSO[,9],Err.ridge[,9],Err.elnet[,9]))
# 
# # Test False Negative (FN) - 0.5n
# err.test.fn.pct1 = data.frame(c(rep("RF",S),rep("Radial SVM",S),rep("Logistic",S),
#                                 rep("Logistic LASSO",S),rep("Logistic Ridge",S),rep("Logistic Elnet",S)), 
#                               c(Err.rf[,10],Err.svm[,10],Err.logistic[,10],
#                                 Err.LASSO[,10],Err.ridge[,10],Err.elnet[,10]))

#####################
# 0.9n Training Size
err.train.pct2 = data.frame(c(rep("RF",S),rep("Radial SVM",S),rep("Logistic",S),
                              rep("Logistic LASSO",S),
                              rep("Logistic Ridge",S),rep("Logistic Elnet",S)), 
                            c(Err.rf[,2],Err.svm[,2],Err.logistic[,2],
                              Err.LASSO[,2],Err.ridge[,2],Err.elnet[,2]))

err.test.pct2  = data.frame(c(rep("RF",S),rep("Radial SVM",S),rep("Logistic",S),
                              rep("Logistic LASSO",S),
                              rep("Logistic Ridge",S),rep("Logistic Elnet",S)), 
                            c(Err.rf[,4],Err.svm[,4],Err.logistic[,4],
                              Err.LASSO[,4],Err.ridge[,4],Err.elnet[,4]))

err.minCV.pct2 = data.frame(c(rep("RF",S),rep("Radial SVM",S),rep("Logistic",S),
                              rep("Logistic LASSO",S),
                              rep("Logistic Ridge",S),rep("Logistic Elnet",S)), 
                            c(Err.rf[,6],Err.svm[,6],Err.logistic[,6],
                              Err.LASSO[,6],Err.ridge[,6],Err.elnet[,6]))

# AUCs
auc.train.pct2 = data.frame(c(rep("RF",S),rep("Radial SVM",S),rep("Logistic",S),
                              rep("Logistic LASSO",S),
                              rep("Logistic Ridge",S),rep("Logistic Elnet",S)), 
                            c(AUC.rf[,3],AUC.svm[,3],AUC.logistic[,3],
                              AUC.LASSO[,3],AUC.ridge[,3],AUC.elnet[,3]))

auc.test.pct2  = data.frame(c(rep("RF",S),rep("Radial SVM",S),rep("Logistic",S),
                              rep("Logistic LASSO",S),
                              rep("Logistic Ridge",S),rep("Logistic Elnet",S)), 
                            c(AUC.rf[,4],AUC.svm[,4],AUC.logistic[,4],
                              AUC.LASSO[,4],AUC.ridge[,4],AUC.elnet[,4]))

# Train False Positive (FP) - 0.9n
err.train.fp.pct2 = data.frame(c(rep("RF",S),rep("Radial SVM",S),
                                 rep("Logistic",S),
                                 rep("Logistic LASSO",S),
                                 rep("Logistic Ridge",S),
                                 rep("Logistic Elnet",S)), 
                               c(Err.rf[,11],Err.svm[,11],Err.logistic[,11],
                                 Err.LASSO[,11],Err.ridge[,11],Err.elnet[,11]))

# Train False Negative (FN) - 0.9n
err.train.fn.pct2 = data.frame(c(rep("RF",S),rep("Radial SVM",S),
                                 rep("Logistic",S),
                                 rep("Logistic LASSO",S),
                                 rep("Logistic Ridge",S),
                                 rep("Logistic Elnet",S)), 
                               c(Err.rf[,12],Err.svm[,12],Err.logistic[,12],
                                 Err.LASSO[,12],Err.ridge[,12],Err.elnet[,12]))

# Test False Positive (FP) - 0.9n
err.test.fp.pct2 = data.frame(c(rep("RF",S),rep("Radial SVM",S),
                                rep("Logistic",S),
                                rep("Logistic LASSO",S),
                                rep("Logistic Ridge",S),
                                rep("Logistic Elnet",S)), 
                              c(Err.rf[,13],Err.svm[,13],Err.logistic[,13],
                                Err.LASSO[,13],Err.ridge[,13],Err.elnet[,13]))

# Test False Negative (FN) - 0.9n
err.test.fn.pct2 = data.frame(c(rep("RF",S),rep("Radial SVM",S),
                                rep("Logistic",S),
                                rep("Logistic LASSO",S),
                                rep("Logistic Ridge",S),
                                rep("Logistic Elnet",S)), 
                              c(Err.rf[,14],Err.svm[,14],Err.logistic[,14],
                                Err.LASSO[,14],Err.ridge[,14],Err.elnet[,14]))
# (Hide)
#####
# colnames(err.train.pct1)    =   c("method","err")
# colnames(err.test.pct1)     =   c("method","err")
# colnames(err.minCV.pct1)    =   c("method","err")
#####

colnames(err.train.pct2)    =   c("method","err")
colnames(err.test.pct2)     =   c("method","err")
colnames(err.minCV.pct2)    =   c("method","err")

# (Hide)
#####
# colnames(err.train.fp.pct1) =   c("method","err")
# colnames(err.train.fn.pct1) =   c("method","err")
# colnames(err.test.fp.pct1)  =   c("method","err")
# colnames(err.test.fn.pct1)  =   c("method","err")
#####

colnames(err.train.fp.pct2) =   c("method","err")
colnames(err.train.fn.pct2) =   c("method","err")
colnames(err.test.fp.pct2)  =   c("method","err")
colnames(err.test.fn.pct2)  =   c("method","err")

colnames(auc.train.pct2)    =   c("method","err")
colnames(auc.test.pct2)     =   c("method","err")

# Saving Files
# (Hide)
#####
# write.csv(err.train.pct1, "err.train.pct1.csv", row.names = FALSE)
# write.csv(err.test.pct1,  "err.test.pct1.csv",  row.names = FALSE)
# write.csv(err.minCV.pct1, "err.minCV.pct1.csv", row.names = FALSE)
#####

write.csv(err.train.pct2, "err.train.pct2.csv", row.names = FALSE)
write.csv(err.test.pct2,  "err.test.pct2.csv",  row.names = FALSE)
write.csv(err.minCV.pct2, "err.minCV.pct2.csv", row.names = FALSE)

# (Hide)
#####
# write.csv(err.train.fp.pct1, "err.train.fp.pct1.csv", row.names = FALSE)
# write.csv(err.train.fn.pct1, "err.train.fn.pct1.csv", row.names = FALSE)
# write.csv(err.test.fp.pct1, "err.test.fp.pct1.csv", row.names = FALSE)
# write.csv(err.test.fn.pct1, "err.test.fn.pct1.csv", row.names = FALSE)
#####

write.csv(err.train.fp.pct2, "err.train.fp.pct2.csv", row.names = FALSE)
write.csv(err.train.fn.pct2, "err.train.fn.pct2.csv", row.names = FALSE)
write.csv(err.test.fp.pct2, "err.test.fp.pct2.csv", row.names = FALSE)
write.csv(err.test.fn.pct2, "err.test.fn.pct2.csv", row.names = FALSE)

# Saving AUC files
write.csv(auc.train.pct2, "auc.train.pct2.csv", row.names = FALSE)
write.csv(auc.test.pct2, "auc.test.pct2.csv", row.names = FALSE)

# Checkpoint 2
save.image(file = 'Checkpoint 2 - Errors and AUCs.RData')



###############################################################################
# V. Boxplots
###############################################################################



#####
# Method 1: Generate 6 boxplots with 6 legends for each nlearn
# p1 = ggplot(err.train.pct1)   +   aes(x=method, y = err, fill=method) +   geom_boxplot()  +
#   ggtitle("0.5n train errors") +
#   theme( axis.title.x = element_text(size =12, face  = "bold", family = "Courier"),
#          plot.title   = element_text(size =12, family= "Courier"), 
#          axis.title.y = element_text(size =12, face  = "bold", family = "Courier"), 
#          axis.text.x  = element_text(angle=45, hjust =  1, size=10, face="bold", family="Courier"), 
#          axis.text.y  = element_text(angle=45, vjust =0.7, size=10, face="bold", family="Courier"))+
#   ylim(0, 1)  
# 
# p2 = ggplot(err.test.pct1)   +    aes(x=method, y = err, fill=method) +   geom_boxplot()  +
#   ggtitle("0.5n test errors") +
#   theme( axis.title.x = element_text(size =12, face  = "bold", family = "Courier"),
#          plot.title   = element_text(size =12, family= "Courier"), 
#          axis.title.y = element_text(size =12, face  = "bold", family = "Courier"), 
#          axis.text.x  = element_text(angle=45, hjust =  1, size=10, face="bold", family="Courier"), 
#          axis.text.y  = element_text(angle=45, vjust =0.7, size=10, face="bold", family="Courier"))+
#   ylim(0, 1)  
# 
# p3 = ggplot(err.minCV.pct1)   +     aes(x=method, y = err, fill=method) +   geom_boxplot()  +  
#   ggtitle("0.5n min CV errors") +
#   theme( axis.title.x = element_text(size =12, face  = "bold", family = "Courier"),
#          plot.title   = element_text(size =12, family= "Courier"), 
#          axis.title.y = element_text(size =12, face  = "bold", family = "Courier"), 
#          axis.text.x  = element_text(angle=45, hjust =  1, size=10, face="bold", family="Courier"), 
#          axis.text.y  = element_text(angle=45, vjust =0.7, size=10, face="bold", family="Courier"))+
#   ylim(0, 1)  
#####

p4 = ggplot(err.train.pct2) + aes(x=method, y = err, fill = method) +
     geom_boxplot() + ggtitle("0.9n train errors") +
     theme(axis.title.x = element_text(size = 12, face  = "bold",
                                     family = "Courier"),
         plot.title   = element_text(size = 12, family= "Courier"), 
         axis.title.y = element_text(size = 12, face  = "bold",
                                     family = "Courier"), 
         axis.text.x  = element_text(angle = 45, hjust =  1, size = 10,
                                     face = "bold", family = "Courier"), 
         axis.text.y  = element_text(angle = 45, vjust = 0.7, size = 10,
                                     face = "bold", family = "Courier")) +
  ylim(0, 1)  

p5 = ggplot(err.test.pct2) + aes(x=method, y = err, fill = method) +
     geom_boxplot()  +  ggtitle("0.9n test errors") +
     theme(axis.title.x = element_text(size = 12, face  = "bold",
                                            family = "Courier"),
         plot.title   = element_text(size = 12, family = "Courier"), 
         axis.title.y = element_text(size = 12, face  = "bold",
                                     family = "Courier"), 
         axis.text.x  = element_text(angle = 45, hjust =  1, size = 10,
                                     face = "bold", family = "Courier"), 
         axis.text.y  = element_text(angle = 45, vjust = 0.7, size = 10,
                                     face = "bold", family = "Courier")) +
  ylim(0, 1)  

p6 = ggplot(err.minCV.pct2) + aes(x = method, y = err, fill = method) +
     geom_boxplot() + ggtitle("0.9n min CV errors") +
     theme(axis.title.x = element_text(size = 12, face  = "bold",
                                       family = "Courier"),
         plot.title   = element_text(size = 12, family = "Courier"), 
         axis.title.y = element_text(size = 12, face  = "bold",
                                     family = "Courier"), 
         axis.text.x  = element_text(angle = 45, hjust =  1, size = 10,
                                     face = "bold", family = "Courier"), 
         axis.text.y  = element_text(angle = 45, vjust = 0.7, size = 10,
                                     face = "bold", family = "Courier")) +
  ylim(0, 1)  

#####
# p7 = ggplot(err.train.fp.pct1)   +     aes(x=method, y = err, fill=method) +   geom_boxplot()  + 
#   ggtitle("0.5n train fp errors") +
#   theme( axis.title.x = element_text(size =12, face  = "bold", family = "Courier"),
#          plot.title   = element_text(size =12, family= "Courier"), 
#          axis.title.y = element_text(size =12, face  = "bold", family = "Courier"), 
#          axis.text.x  = element_text(angle=45, hjust =  1, size=10, face="bold", family="Courier"), 
#          axis.text.y  = element_text(angle=45, vjust =0.7, size=10, face="bold", family="Courier"))+
#   ylim(0, 1)  
# 
# p8 = ggplot(err.train.fn.pct1)   +     aes(x=method, y = err, fill=method) +   geom_boxplot()  + 
#   ggtitle("0.5n train fn errors") +
#   theme( axis.title.x = element_text(size =12, face  = "bold", family = "Courier"),
#          plot.title   = element_text(size =12, family= "Courier"), 
#          axis.title.y = element_text(size =12, face  = "bold", family = "Courier"), 
#          axis.text.x  = element_text(angle=45, hjust =  1, size=10, face="bold", family="Courier"), 
#          axis.text.y  = element_text(angle=45, vjust =0.7, size=10, face="bold", family="Courier"))+
#   ylim(0, 1)  
# 
# p9 = ggplot(err.test.fp.pct1)   +     aes(x=method, y = err, fill=method) +   geom_boxplot()  + 
#   ggtitle("0.5n test fp errors") +
#   theme( axis.title.x = element_text(size =12, face  = "bold", family = "Courier"),
#          plot.title   = element_text(size =12, family= "Courier"), 
#          axis.title.y = element_text(size =12, face  = "bold", family = "Courier"), 
#          axis.text.x  = element_text(angle=45, hjust =  1, size=10, face="bold", family="Courier"), 
#          axis.text.y  = element_text(angle=45, vjust =0.7, size=10, face="bold", family="Courier"))+
#   ylim(0, 1)  
# 
# p10 = ggplot(err.test.fn.pct1)   +     aes(x=method, y = err, fill=method) +   geom_boxplot()  + 
#   ggtitle("0.5n test fn errors") +
#   theme( axis.title.x = element_text(size =12, face  = "bold", family = "Courier"),
#          plot.title   = element_text(size =12, family= "Courier"), 
#          axis.title.y = element_text(size =12, face  = "bold", family = "Courier"), 
#          axis.text.x  = element_text(angle=45, hjust =  1, size=10, face="bold", family="Courier"), 
#          axis.text.y  = element_text(angle=45, vjust =0.7, size=10, face="bold", family="Courier"))+
#   ylim(0, 1)  
#####

p11 = ggplot(err.train.fp.pct2) + aes(x = method, y = err, fill = method) +
      geom_boxplot() + ggtitle("0.9n train fp errors") +
      theme(axis.title.x = element_text(size = 12, face  = "bold",
                                     family = "Courier"),
         plot.title   = element_text(size = 12, family= "Courier"), 
         axis.title.y = element_text(size = 12, face  = "bold",
                                     family = "Courier"), 
         axis.text.x  = element_text(angle = 45, hjust =  1, size = 10,
                                     face = "bold", family = "Courier"), 
         axis.text.y  = element_text(angle = 45, vjust = 0.7, size = 10,
                                     face = "bold", family = "Courier")) +
  ylim(0, 1)  

p12 = ggplot(err.train.fn.pct2) + aes(x = method, y = err, fill = method) +
      geom_boxplot() + ggtitle("0.9n train fn errors") +
      theme(axis.title.x = element_text(size = 12, face  = "bold",
                                         family = "Courier"),
         plot.title   = element_text(size = 12, family = "Courier"), 
         axis.title.y = element_text(size = 12, face  = "bold",
                                     family = "Courier"), 
         axis.text.x  = element_text(angle = 45, hjust =  1, size = 10,
                                     face = "bold", family = "Courier"), 
         axis.text.y  = element_text(angle = 45, vjust = 0.7, size = 10,
                                     face = "bold", family = "Courier")) +
  ylim(0, 1)  

p13 = ggplot(err.test.fp.pct2) + aes(x = method, y = err, fill = method) +
      geom_boxplot() + ggtitle("0.9n test fp errors") +
      theme(axis.title.x = element_text(size = 12, face  = "bold",
                                         family = "Courier"),
         plot.title   = element_text(size = 12, family = "Courier"), 
         axis.title.y = element_text(size = 12, face  = "bold",
                                     family = "Courier"), 
         axis.text.x  = element_text(angle = 45, hjust =  1, size = 10,
                                     face = "bold", family = "Courier"), 
         axis.text.y  = element_text(angle = 45, vjust = 0.7, size = 10,
                                     face = "bold", family = "Courier")) +
  ylim(0, 1)  

p14 = ggplot(err.test.fn.pct2) + aes(x = method, y = err, fill = method) +
      geom_boxplot() + ggtitle("0.9n test fp errors") + 
      theme( axis.title.x = element_text(size = 12, face  = "bold",
                                     family = "Courier"),
         plot.title   = element_text(size = 12, family = "Courier"), 
         axis.title.y = element_text(size = 12, face  = "bold",
                                     family = "Courier"), 
         axis.text.x  = element_text(angle = 45, hjust =  1, size = 10,
                                     face = "bold", family = "Courier"), 
         axis.text.y  = element_text(angle = 45, vjust = 0.7, size = 10,
                                     face = "bold", family = "Courier")) +
  ylim(0, 1)  

# grid.arrange(p1,  p7,  p8, p2,  p9, p10, ncol=3)
# grid.arrange(p4, p11, p12, p5, p13, p14, ncol=3)

grid.arrange(p4, p5, p11, p12, p13, p14, ncol = 3)

# (Hide)
# #####################################################
# # Method 2: Generate 6 boxplots with only one legend
# 
# # 0.5n
# giant_df01 = data.frame(c(rep("0.5n Train Errors",6*S), 
#                         rep("0.5n Train FP Errors",6*S), 
#                         rep("0.5n Train FN Errors",6*S),
#                         rep("0.5n Test Errors",6*S), 
#                         rep("0.5n Test FP Errors",6*S), 
#                         rep("0.5n Test FN Errors",6*S)), 
#                         
#                       rbind(err.train.pct1, 
#                             err.train.fp.pct1, 
#                             err.train.fn.pct1, 
#                             err.test.pct1,
#                             err.test.fp.pct1,
#                             err.test.fn.pct1) 
#                       ) 
# 
# colnames(giant_df01)    =     c("error_type","method","err")
# giant_df01$error_type = factor(giant_df01$error_type, 
#                              levels = c("0.5n Train Errors", 
#                                         "0.5n Train FP Errors", 
#                                         "0.5n Train FN Errors",
#                                         "0.5n Test Errors",
#                                         "0.5n Test FP Errors", 
#                                         "0.5n Test FN Errors"))
# 
# giant_p01 = ggplot(giant_df01)   +     
#   aes(x = method, y = err, fill = method) +   
#   geom_boxplot()  + 
#   facet_wrap(~ error_type, ncol = 3) + 
#   ggtitle("Boxplots of Error Rates (train size = 0.5n, 50 samples)") +
#   labs(title = 'Boxplots of Error Rates (train size = 0.5n, 50 samples)',
#        x = "Method", y = "Error Rate") +
#   theme( axis.title.x = element_text(face = "bold"),
#          plot.title   = element_text(), 
#          axis.title.y = element_text(face = "bold"), 
#          axis.text.x  = element_text(angle = 20, hjust =  1), 
#          axis.text.y  = element_text(angle = 20, vjust = 0.7)) +
#   ylim(0, 0.6)  
# 
# giant_p01
# 
# # 0.9n
# giant_df02 = data.frame(c(rep("0.9n Train Errors",6*S), 
#                           rep("0.9n Train FP Errors",6*S), 
#                           rep("0.9n Train FN Errors",6*S),
#                           rep("0.9n Test Errors",6*S), 
#                           rep("0.9n Test FP Errors",6*S), 
#                           rep("0.9n Test FN Errors",6*S)), 
#                         
#                         rbind(err.train.pct1, 
#                               err.train.fp.pct2, 
#                               err.train.fn.pct2, 
#                               err.test.pct2,
#                               err.test.fp.pct2,
#                               err.test.fn.pct2) 
# ) 
# 
# colnames(giant_df02)    =     c("error_type","method","err")
# giant_df02$error_type = factor(giant_df02$error_type, 
#                                levels = c("0.9n Train Errors", 
#                                           "0.9n Train FP Errors", 
#                                           "0.9n Train FN Errors",
#                                           "0.9n Test Errors",
#                                           "0.9n Test FP Errors", 
#                                           "0.9n Test FN Errors"))
# 
# giant_p02 = ggplot(giant_df02)   +     
#   aes(x = method, y = err, fill = method) +   
#   geom_boxplot()  + 
#   facet_wrap(~ error_type, ncol = 3) + 
#   ggtitle("Boxplots of Error Rates (train size = 0.9n, 50 samples)") +
#   labs(title = 'Boxplots of Error Rates (train size = 0.9n, 50 samples)',
#        x = "Method", y = "Error Rate") +
#   theme( axis.title.x = element_text(face = "bold"),
#          plot.title   = element_text(), 
#          axis.title.y = element_text(face = "bold"), 
#          axis.text.x  = element_text(angle = 20, hjust =  1), 
#          axis.text.y  = element_text(angle = 20, vjust = 0.7)) +
#   ylim(0, 0.6)  
# 
# giant_p02
# 
# #####################################################
# # Method 3: Generate 4 boxplots with only one legend
# 
# giant_df = data.frame(c(rep("0.5n Train Errors",6*S),
#                         rep("0.5n Test Errors",6*S),
#                         rep("0.9n Train Errors",6*S),
#                         rep("0.9n Test Errors",6*S)),
# 
#                       rbind(err.train.pct1,
#                             err.test.pct1,
#                             err.train.pct2,
#                             err.test.pct2)
# )
# 
# colnames(giant_df)    =     c("error_type","method","err")
# giant_df$error_type = factor(giant_df$error_type,
#                              levels = c("0.5n Train Errors",
#                                       "0.5n Test Errors",
#                                       "0.9n Train Errors",
#                                       "0.9n Test Errors"))
# 
# giant_p = ggplot(giant_df)   +
#   aes(x = method, y = err, fill = method) +
#   geom_boxplot()  +
#   facet_wrap(~ error_type, ncol = 2) + 
#   ggtitle("Boxplots of Error Rates (train size = 0.5n, 0.9n, 50 samples)") +
#   labs(title = 'Boxplots of Error Rates (train size = 0.5n, 0.9n, 50 samples)',
#        x = "Method", y = "Error Rate") +
#   theme( axis.title.x = element_text(face = "bold"),
#          plot.title   = element_text(), 
#          axis.title.y = element_text(face = "bold"), 
#          axis.text.x  = element_text(angle = 20, hjust =  1), 
#          axis.text.y  = element_text(angle = 20, vjust = 0.7)) +
#   ylim(0, 0.6)
# 
# giant_p

######################################
# Q3(b). Boxplot of AUCs (Train & Test)

# Combined AUC / Error Rate Boxplots

giant_df_new = data.frame(c(rep("0.9n Train AUCs",6*S),
                        rep("0.9n Test AUCs",6*S),
                        rep("0.9n Train Errors",6*S),
                        rep("0.9n Test Errors",6*S)),
                      
                      rbind(auc.train.pct2,
                            auc.test.pct2,
                            err.train.pct2,
                            err.test.pct2)
)

colnames(giant_df_new)    =     c("error_type","method","err")
giant_df_new$error_type = factor(giant_df_new$error_type,
                             levels = c("0.9n Train AUCs",
                                        "0.9n Test AUCs",
                                        "0.9n Train Errors",
                                        "0.9n Test Errors"))

giant_plot = ggplot(giant_df_new)   +
  aes(x = method, y = err, fill = method) +
  geom_boxplot()  +
  facet_wrap(~ error_type, ncol = 2) + 
  ggtitle("Boxplots of AUCs and Error Rates (train size = 0.9n, 50 samples)") +
  labs(title =
         expression('Boxplots of AUCs and Error Rates (n'[train]*' = 0.9n, 50 samples)'),
       x = "Method", y = "Error Rate                                                                                          AUC") +
  theme( axis.title.x = element_text(face = "bold"),
         plot.title   = element_text(), 
         axis.title.y = element_text(face = "bold"), 
         axis.text.x  = element_text(angle = 20, hjust =  1), 
         axis.text.y  = element_text(angle = 20, vjust = 0.7)) +
  ylim(0, 1)

giant_plot

# (Hide)
#####
# # Individual plots
# p_01 = ggplot(auc.train.pct2) + aes(x=method, y = err, fill=method) + geom_boxplot() + 
#   ggtitle("0.9n train AUCs") + 
#   theme( axis.title.x = element_text(size  = 12, face  = "bold", family = "Courier"),
#          plot.title   = element_text(size  = 12, family= "Courier"), 
#          axis.title.y = element_text(size  = 12, face  = "bold", family = "Courier"), 
#          axis.text.x  = element_text(angle = 45, hjust =  1, size = 10, face = "bold", family = "Courier"), 
#          axis.text.y  = element_text(angle = 45, vjust = 0.7,size = 10, face = "bold", family = "Courier")) +
#   ylim(0, 1)  
# 
# p_02 = ggplot(auc.test.pct2) + aes(x=method, y = err, fill=method) + geom_boxplot() +
#   ggtitle("0.9n test AUCs") +
#   theme( axis.title.x = element_text(size  = 12, face  = "bold", family = "Courier"),
#          plot.title   = element_text(size  = 12, family= "Courier"), 
#          axis.title.y = element_text(size  = 12, face  = "bold", family = "Courier"), 
#          axis.text.x  = element_text(angle = 45, hjust =  1, size = 10, face = "bold", family = "Courier"), 
#          axis.text.y  = element_text(angle = 45, vjust = 0.7,size = 10, face = "bold", family = "Courier")) +
#   ylim(0, 1) 
# 
# grid.arrange(p_01, p_02, ncol = 2)
#####

# Two AUC Boxplots only
giant_df_new02 = data.frame(c(rep("0.9n Train AUCs",6*S),
                            rep("0.9n Test AUCs",6*S)),
                            
                          rbind(auc.train.pct2,
                                auc.test.pct2)
)

colnames(giant_df_new02)    =     c("error_type","method","err")
giant_df_new02$error_type = factor(giant_df_new02$error_type,
                                 levels = c("0.9n Train AUCs",
                                            "0.9n Test AUCs"))

aucs.df.boxplot = ggplot(giant_df_new02) + 
  aes(x = method, y = err, fill = method) + 
  geom_boxplot() + 
  facet_wrap(~ error_type, ncol = 2) + 
  labs(title =
         expression('Boxplots of AUCs via Six Methods (n'[train]*' = 0.9n, 50 samples)'),
       x = "Method", y = "AUC",fill = "Method") +
  theme( axis.title.x = element_text(face = "bold"),
       plot.title   = element_text(), 
       axis.title.y = element_text(face = "bold"), 
       axis.text.x  = element_text(angle = 20, hjust =  1), 
       axis.text.y  = element_text(angle = 20, vjust = 0.7)) + ylim(0, 1)

aucs.df.boxplot

# Checkpoint 3
save.image(file = 'Checkpoint 3 - Boxplots.RData')



###############################################################################
# VI. 10-fold Cross-validation (CV) Curves & Model Performance
###############################################################################



###########################################################
# 3(c). 10-fold CV curves for LASSO, Ridge, Elnet, and SVM

s = 50                                                                           # Sampling order
set.seed(1)

for (j in 1:2) {
  
  n.train      =  floor(n*learn.pct[j])
  n.test       =  n-n.train
  trainSet     =  random_order[s,][1:n.train]
  testSet      =  random_order[s,][(1+n.train):n] 
  X.train      =  X[trainSet, ]
  y.train      =  y[trainSet]
  X.test       =  X[testSet, ]
  y.test       =  y[testSet]
  y.os.train   =  y.train                                                        # Initialize the over-sampled set to train models
  X.os.train   =  X.train                                                        # Initialize the over-sampled set to train models
  
  ######################################################
  # To account for imbalance, data will be over-sampled
  # (w/ replacement) to become balanced
  imbalance     =     FALSE   
  if (imbalance == TRUE) {
    index.yis0      =      which(y.train == 0)                                   # Identify index of points with label 0
    index.yis1      =      which(y.train == 1)                                   # Identify index of points with label 1
    n.train.1       =      length(index.yis1)
    n.train.0       =      length(index.yis0)
    if (n.train.1 > n.train.0) {                                                 # Needing more 0s in training set (Over-sampling with replacement)
      more.train    =      sample(index.yis0, size = n.train.1-n.train.0,
                                  replace = TRUE)
    }         else {                                                             # Needing more 1s in training set (Over-sampling with replacement)        
      more.train    =      sample(index.yis1, size = n.train.0-n.train.1,
                                  replace = TRUE)
    }
    
    ##########################################################
    # The code below CORRECTLY over-samples the training set
    # and stores into y.train and X.train
    y.os.train        =       as.factor(c(y.train, y.train[more.train])-1) 
    X.os.train        =       rbind2(X.train, X.train[more.train,])    
  }

  os.train.data           =   data.frame(X.os.train, as.factor(y.os.train))
  train.data              =   data.frame(X.train, as.factor(y.train))
  test.data               =   data.frame(X.test, as.factor(y.test))
  names(os.train.data)[89]=   "target"
  names(train.data)[89]   =   "target"
  names(test.data)[89]    =   "target"
  
  #m             =    25
  m             =    50
  
  ########################################
  # Logistic LASSO - CV curve and runtime
  ########################################
  
  LASSO.CV      =    cv.glmnet(X.os.train, y.os.train, family = "binomial",
                               alpha = 1, 
                               intercept = TRUE, standardize = FALSE,  
                               nfolds = 10, type.measure = "class")
  lam.LASSO     =    exp(seq(log(max(LASSO.CV$lambda)),log(0.00001), 
                             (log(0.00001) - log(max(LASSO.CV$lambda)))/(m-1)))
  
  ptm           =    proc.time()
  LASSO.CV      =    cv.glmnet(X.os.train, y.os.train, lambda = lam.LASSO, 
                               family = "binomial", alpha = 1, intercept = TRUE, 
                               standardize = FALSE, nfolds = 10,
                               type.measure = "class")
  ptm           =    proc.time() - ptm
  time.LASSO.CV =    ptm["elapsed"] 
  
  ptm           =    proc.time()
  LASSO.fit     =    glmnet(X.os.train, y.os.train, lambda = LASSO.CV$lambda, 
                            family = "binomial", alpha = 1, 
                            intercept = TRUE, standardize = FALSE)
  ptm           =    proc.time() - ptm
  time.LASSO.fit=    ptm["elapsed"] 
  
  LASSO.fit.0   =    glmnet(X.os.train, y.os.train, lambda = 0, 
                            family = "binomial", alpha = 1, 
                            intercept = TRUE, standardize = FALSE)
  
  n.lambdas     =    dim(LASSO.fit$beta)[2]
  LASSO.beta.ratio    =    rep(0, n.lambdas)
  for (i in 1:n.lambdas) {
    LASSO.beta.ratio[i] =
      sum(abs(LASSO.fit$beta[,i]))/sum(abs(LASSO.fit.0$beta))
  }
  
  ########################################
  # Logistic Ridge - CV curve and runtime
  ########################################
  
  ridge.CV      =    cv.glmnet(X.os.train, y.os.train, family = "binomial",
                               alpha = 0, 
                               intercept = TRUE, standardize = FALSE, 
                               nfolds = 10, type.measure = "class")
  lam.ridge     =    exp(seq(log(max(ridge.CV$lambda)),log(0.00001), 
                             -(log(max(ridge.CV$lambda))-log(0.00001))/(m-1)))
  
  ptm           =    proc.time()
  ridge.CV      =    cv.glmnet(X.os.train, y.os.train, lambda = lam.ridge, 
                               family = "binomial", alpha = 0, intercept = TRUE, 
                               standardize = FALSE, nfolds = 10,
                               type.measure = "class")
  ptm           =    proc.time() - ptm
  time.ridge.CV =    ptm["elapsed"] 
  
  ptm           =    proc.time()
  ridge.fit     =    glmnet(X.os.train, y.os.train, lambda = ridge.CV$lambda, 
                            family = "binomial", alpha = 0,  
                            intercept = TRUE, standardize = FALSE)
  ptm           =    proc.time() - ptm
  time.ridge.fit=    ptm["elapsed"] 
  
  ridge.fit.0   =    glmnet(X.os.train, y.os.train, lambda = 0, 
                            family = "binomial", alpha = 0, 
                            intercept = TRUE, standardize = FALSE)
  
  n.lambdas     =    dim(ridge.fit$beta)[2]
  ridge.beta.ratio    =    rep(0, n.lambdas)
  for (i in 1:n.lambdas) {
    ridge.beta.ratio[i] =
      sqrt(sum((ridge.fit$beta[,i])^2)/sum((ridge.fit.0$beta)^2))
  }
  
  ########################################
  # Logistic Elnet - CV curve and runtime
  ########################################
  
  elnet.CV      =    cv.glmnet(X.os.train, y.os.train, family = "binomial",
                               alpha = 0.5, 
                               intercept = TRUE, standardize = FALSE,  
                               nfolds = 10, type.measure = "class")
  lam.elnet     =    exp(seq(log(max(elnet.CV$lambda)),log(0.00001), 
                             (log(0.00001) - log(max(elnet.CV$lambda)))/(m-1)))
  
  ptm           =    proc.time()
  elnet.CV      =    cv.glmnet(X.os.train, y.os.train, lambda = lam.elnet, 
                               family = "binomial", alpha = 0.5,
                               intercept = TRUE, 
                               standardize = FALSE, nfolds = 10,
                               type.measure = "class")
  ptm           =    proc.time() - ptm
  time.elnet.CV =    ptm["elapsed"] 
  
  ptm           =    proc.time()
  elnet.fit     =    glmnet(X.os.train, y.os.train, lambda = elnet.CV$lambda, 
                            family = "binomial", alpha = 0.5, 
                            intercept = TRUE, standardize = FALSE)
  ptm           =    proc.time() - ptm
  time.elnet.fit=    ptm["elapsed"] 
  
  elnet.fit.0   =    glmnet(X.os.train, y.os.train, lambda = 0, 
                            family = "binomial", alpha = 0.5, 
                            intercept = TRUE, standardize = FALSE)
  
  n.lambdas     =    dim(elnet.fit$beta)[2]
  elnet.beta.ratio    =    rep(0, n.lambdas)
  for (i in 1:n.lambdas) {
    elnet.beta.ratio[i] = sum(abs(elnet.fit$beta[,i]))/sum(abs(elnet.fit.0$beta))
  }
  
  ####################################################################
  # 3(c). Plotting CV curves for LASSO, Ridge, and Elnet on one plot
  
  error           =     data.frame(c(rep("LASSO", length(LASSO.beta.ratio)), 
                                    rep("Ridge", length(ridge.beta.ratio)),
                                    rep("Elnet", length(elnet.beta.ratio))), 
                                  c(LASSO.beta.ratio, ridge.beta.ratio,
                                    elnet.beta.ratio),
                                  c(LASSO.CV$cvm, ridge.CV$cvm, elnet.CV$cvm),
                                  c(LASSO.CV$cvsd, ridge.CV$cvsd,
                                    elnet.CV$cvsd))
  colnames(error) =     c("Method", "Ratio", "CV", "SD")
  
  error.plot      =     ggplot(error, aes(x = Ratio, y = CV, color = Method)) +
                        geom_line(size = 1) 
  error.plot      =     error.plot  +
                        theme(legend.text = element_text(colour = "black")) 
  error.plot      =     error.plot  + geom_pointrange(aes(ymin = CV-SD,
                                                          ymax = CV+SD),
                                                      shape = 15)
  error.plot      =     error.plot  + theme(legend.title = element_blank()) 
  error.plot      =     error.plot  + scale_color_discrete(breaks =
                                                             c("LASSO", "Ridge",
                                                               "Elnet"))
  error.plot      =     error.plot  + theme(axis.title.x = element_text(),
                                          axis.text.x  =
                                            element_text(angle = 0,vjust = 0.5),
                                          axis.text.y  =
                                            element_text(angle = 0,vjust = 0.5)) 
  error.plot      =     error.plot  + theme(plot.title =
                                              element_text(hjust = 0.5,
                                                           vjust = -10))
  error.plot      =     error.plot  + ggtitle((sprintf("%.1fn CV Error Curves \n LASSO.CV: %0.3f (sec), LASSO.fit: %0.3f (sec) \n Ridge.CV: %0.3f (sec), Ridge.fit: %0.3f (sec) \n Elnet.CV: %0.3f (sec), Elnet.fit: %0.3f (sec) \n \n \n",learn.pct[j],time.LASSO.CV,time.LASSO.fit,time.ridge.CV,time.ridge.fit,time.elnet.CV,time.elnet.fit))) 
  
  if (j == 1) {
    error.plot.1 =   error.plot 
    time.LASSO.1 =   time.LASSO.CV + time.LASSO.fit    
    time.ridge.1 =   time.ridge.CV + time.ridge.fit
    time.elnet.1 =   time.elnet.CV + time.elnet.fit
  } else {
    error.plot.2 =   error.plot 
    time.LASSO.2 =   time.LASSO.CV + time.LASSO.fit
    time.ridge.2 =   time.ridge.CV + time.ridge.fit
    time.elnet.2 =   time.elnet.CV + time.elnet.fit
  }
  
  ####################################
  # Radial SVM - CV error and runtime
  ####################################
  
  ptm         =   proc.time()
  tune.svm    =   tune(svm, target ~ ., data = train.data, kernel = "radial", 
                       ranges = list(cost = 10^seq(-2,2,length.out = 5), 
                                     gamma = 10^seq(-2,2,length.out = 5) ))
  ptm         =   proc.time() - ptm
  time.SVM.CV =   ptm["elapsed"] 
  
  ptm         =   proc.time()
  SVM.fit     =   svm(target~., data = train.data, kernel = "radial", 
                      cost=tune.svm$performances$cost, 
                      gamma=tune.svm$performances$gamma)
  ptm         =   proc.time() - ptm
  time.SVM.fit=   ptm["elapsed"] 
  
  ##################################
  # Generating SVM CV error heatmap
  
  svm.df   = data.frame(as.character(tune.svm$performances[,1]), 
                        as.character(tune.svm$performances[,2]), 
                        tune.svm$performances[,3])
  colnames(svm.df) = c("Cost", "Gamma","Error")
  
  svm.heat = ggplot(svm.df, aes(Gamma, Cost, fill = Error)) + 
    geom_tile(colour = "white") + 
    geom_text(aes(label = round(Error, 3))) +
    scale_fill_gradientn(colours = c("yellow", "red")) + 
    ggtitle((sprintf("%.1fn Heatmap \n SVM.CV: %0.3f (sec), SVM.fit: %0.3f (sec)",
                     learn.pct[j],time.SVM.CV,time.SVM.fit))) + 
    theme_classic() 
  
  if (j == 1) {
    svm.heat.1 =   svm.heat 
    time.svm.1 =   time.SVM.CV + time.SVM.fit
  } else {
    svm.heat.2 =   svm.heat
    time.svm.2 =   time.SVM.CV + time.SVM.fit
  }
  
  ##############################################
  # Random Forest - CV runtime and fitting time
  ##############################################
  
  ptm         =   proc.time()
  tune.rf     =   tune(randomForest, target ~ ., data = os.train.data)
  ptm         =   proc.time() - ptm
  time.rf.cv  =   ptm["elapsed"] 
  
  if (j == 1) {
    time.rf.cv.1 =   time.rf.cv 
  } else {
    time.rf.cv.2 =   time.rf.cv 
  }
  
  ptm         =   proc.time()
  rf.fit      =   randomForest(target~., data = os.train.data, mtry = sqrt(p), 
                               importance = TRUE)
  ptm         =   proc.time() - ptm
  time.rf.fit =   ptm["elapsed"] 
  
  if (j == 1) {
    time.rf.fit.1 =   time.rf.fit
  } else {
    time.rf.fit.2 =   time.rf.fit
  }

} 

# CV Error Curves
error.plot.1
error.plot.2

# CV Curves - 1 x 3
par(mfrow = c(1,3))
plot(LASSO.CV)
title("10-fold CV Curve - LASSO", line = 3)
plot(elnet.CV)
title("10-fold CV Curve - Elastic-net", line = 3)
plot(ridge.CV)
title("10-fold CV Curve - Ridge", line = 3)

# CV Curves - 1 x 1
par(mfrow = c(1,1))
# LASSO.CV      =    cv.glmnet(X.os.train, y.os.train, lambda = lam.LASSO, 
#                              family = "binomial", alpha = 1, intercept = TRUE, 
#                              standardize = FALSE, nfolds = 10,
#                              type.measure = "class")    
plot(LASSO.CV)
title("10-fold CV Curve - LASSO", line = 3)
# elnet.CV      =    cv.glmnet(X.os.train, y.os.train, lambda = lam.elnet, 
#                              family = "binomial", alpha = 0.5,
#                              intercept = TRUE, 
#                              standardize = FALSE, nfolds = 10,
#                              type.measure = "class")
plot(elnet.CV)
title("10-fold CV Curve - Elastic-net", line = 3)
# ridge.CV      =    cv.glmnet(X.os.train, y.os.train, lambda = lam.ridge, 
#                              family = "binomial", alpha = 0, intercept = TRUE, 
#                              standardize = FALSE, nfolds = 10,
#                              type.measure = "class")
plot(ridge.CV)
title("10-fold CV Curve - Ridge", line = 3)

# Random Forest runtimes
time.rf.cv.1
time.rf.cv.2
time.rf.fit.1
time.rf.fit.2 

# Radial SVM Heatmap
# svm.heat.1
# svm.heat.2

# Checkpoint 4
save.image(file = 'Checkpoint 4 - CV curves, Heatmaps, and Times.RData')



###############################################################################
# VII. Model Performance Continued & Variable Importance Barplots
###############################################################################



#######################################################################
# 3(d) i. Model Performance vs. Fitting Time for LASSO, Ridge, RF, SVM

# Create performance vs. fitting time data.frame
pt.df = data.frame(c("LASSO", "Ridge", "RF", "SVM", "Elnet"),
                   c(rep("0.9n", 5)),
                   c(time.LASSO.2, time.ridge.2, time.rf.fit.2, time.svm.2,
                     time.elnet.2), 
                   c(Err.LASSO[s,4], Err.ridge[s,4], Err.rf[s,4], Err.svm[s,4],
                     Err.elnet[s,4]))

colnames(pt.df) = c("model", "train_size", "fitting_time", "test_error")
pt.df

write.csv(pt.df, "performance_vs_time.csv", row.names = FALSE)

ggplot(pt.df, aes(x = fitting_time, y = test_error, color = model)) +
  geom_point(size = 5, alpha = 0.7) + 
  facet_wrap(~ train_size, nrow = 2) + 
  labs(x = "Fitting Time (in secs)") + 
  labs(y = "Performance (Test Error Rate)") + 
  ylim(0, 0.6) + 
  ggtitle(expression('Performance vs. Fitting Time (n'[train]*' = 0.9n)'))


###########
# Barplots

########################################################################
# 3(d) ii. Barplots of the estimated coefficients (LASSO, Ridge, Elnet)
#          for variable (parameter) importance, for each nlearn

s = 50
set.seed(1)

for (j in 1:2) {
  
  n.train      =  floor(n*learn.pct[j])
  n.test       =  n-n.train
  trainSet     =  random_order[s,][1:n.train]
  testSet      =  random_order[s,][(1+n.train):n] 
  X.train      =  X[trainSet, ]
  y.train      =  y[trainSet]
  X.test       =  X[testSet, ]
  y.test       =  y[testSet]
  y.os.train   =  y.train
  X.os.train   =  X.train
  
  ######################################################
  # To account for imbalance, data will be over-sampled
  # (w/ replacement) to become balanced
  
  imbalance     =     FALSE   
  if (imbalance == TRUE) {
    index.yis0      =      which(y.train == 0)
    index.yis1      =      which(y.train == 1)
    n.train.1       =      length(index.yis1)
    n.train.0       =      length(index.yis0)
    if (n.train.1 > n.train.0) {
      more.train    =      sample(index.yis0, size = n.train.1-n.train.0,
                                  replace = TRUE)
    }         else {
      more.train    =      sample(index.yis1, size = n.train.0-n.train.1,
                                  replace = TRUE)
    }
    
    ##########################################################
    # The code below CORRECTLY over-samples the training set
    # and stores into y.train and X.train
    
    y.os.train        =       as.factor(c(y.train, y.train[more.train])-1) 
    X.os.train        =       rbind2(X.train, X.train[more.train,])    
  }
  
  os.train.data           =   data.frame(X.os.train, as.factor(y.os.train))
  train.data              =   data.frame(X.train, as.factor(y.train))
  test.data               =   data.frame(X.test, as.factor(y.test))
  names(os.train.data)[89]=   "target"
  names(train.data)[89]   =   "target"
  names(test.data)[89]    =   "target"
  
  #################
  # LASSO Barplots
  #################
  
  LASSO.CV      =    cv.glmnet(X.os.train, y.os.train, family = "binomial",
                               alpha = 1, intercept = TRUE, standardize = FALSE,  
                               nfolds = 10, type.measure = "class")
  lam.LASSO     =    exp(seq(log(max(LASSO.CV$lambda)),log(0.00001), 
                             (log(0.00001) - log(max(LASSO.CV$lambda)))/(m-1)))
  
  LASSO.CV      =    cv.glmnet(X.os.train, y.os.train, lambda = lam.LASSO, 
                               family = "binomial", alpha = 1, intercept = TRUE, 
                               standardize = FALSE, nfolds = 10,
                               type.measure = "class")
  
  LASSO.fit     =    glmnet(X.os.train, y.os.train, lambda = LASSO.CV$lambda.min, 
                            family = "binomial", alpha = 1, 
                            intercept = TRUE, standardize = FALSE)
  
  LASSO.df = data.frame(Predictor=as.vector(row.names(LASSO.fit$beta)), 
                        LASSO.coef=as.vector(LASSO.fit$beta))
  
  if (j == 1) {
    LASSO.bar = ggplot(LASSO.df, aes(x = reorder(Predictor, -LASSO.coef),
                                     y = LASSO.coef, fill = LASSO.coef > 0)) + 
      geom_bar(stat = "identity") + 
      labs(x = "") + 
      labs(y = "Values") + 
      ylim(-0.75, 0.75) + 
      theme(legend.position = "none") + #coord_flip() + 
      ggtitle((sprintf("%.1fn Estimated Coefficients - LASSO",learn.pct[j]))) 
    
    LASSO.bar_20 = top_n(LASSO.df, n = 20, abs(LASSO.coef)) %>% 
      ggplot(., aes(x = reorder(Predictor, -LASSO.coef), y = LASSO.coef,
                    fill = LASSO.coef > 0)) + 
      geom_bar(stat = "identity") + 
      labs(x = "") + 
      labs(y = "Values") + 
      ylim(-0.75, 0.75) + 
      theme(legend.position = "none") + #coord_flip() + 
      ggtitle((sprintf("%.1fn Estimated Coefficients (Top 20) - LASSO",
                       learn.pct[j]))) 
  } else {
    LASSO.bar = ggplot(LASSO.df, aes(x = reorder(Predictor, -LASSO.coef),
                                     y = LASSO.coef, fill = LASSO.coef > 0)) + 
      geom_bar(stat = "identity") + 
      labs(x = "") + 
      labs(y = "Values") + 
      ylim(-0.75, 0.75) + 
      theme(legend.position = "none") + #coord_flip() + 
      ggtitle((sprintf("%.1fn Estimated Coefficients - LASSO",learn.pct[j]))) 
    
    LASSO.bar_20 = top_n(LASSO.df, n = 20, abs(LASSO.coef)) %>% 
      ggplot(., aes(x = reorder(Predictor, -LASSO.coef), y = LASSO.coef,
                    fill = LASSO.coef > 0)) + 
      geom_bar(stat = "identity") + 
      labs(x = "") + 
      labs(y = "Values") + 
      ylim(-0.75, 0.75) + 
      theme(legend.position = "none") + #coord_flip() + 
      ggtitle((sprintf("%.1fn Estimate Coefficients (Top 20) - LASSO",
                       learn.pct[j]))) 
  }
  
  #################
  # Ridge Barplots
  #################
  
  ridge.CV      =    cv.glmnet(X.os.train, y.os.train, family = "binomial",
                               alpha = 0, intercept = TRUE, standardize = FALSE, 
                               nfolds = 10, type.measure = "class")
  lam.ridge     =    exp(seq(log(max(ridge.CV$lambda)),log(0.00001), 
                             -(log(max(ridge.CV$lambda))-log(0.00001))/(m-1)))
  
  ridge.CV      =    cv.glmnet(X.os.train, y.os.train, lambda = lam.ridge, 
                               family = "binomial", alpha = 0, intercept = TRUE, 
                               standardize = FALSE, nfolds = 10,
                               type.measure = "class")
  
  ridge.fit     =    glmnet(X.os.train, y.os.train,
                            lambda = ridge.CV$lambda.min, 
                            family = "binomial", alpha = 0,  
                            intercept = TRUE, standardize = FALSE)
  
  ridge.df = data.frame(Predictor = as.vector(row.names(ridge.fit$beta)), 
                        ridge.coef = as.vector(ridge.fit$beta))
  
  if (j == 1) {
    ridge.bar = ggplot(ridge.df, aes(x = reorder(Predictor, -ridge.coef),
                                     y = ridge.coef, fill = ridge.coef > 0)) + 
      geom_bar(stat = "identity") + 
      labs(x = "") + 
      labs(y = "") + 
      ylim(-0.75, 0.75) + 
      theme(legend.position = "none") + #coord_flip() + 
      ggtitle((sprintf("%.1fn Estimated Coefficients - Ridge",learn.pct[j]))) 
    
    ridge.bar_20 = top_n(ridge.df, n = 20, abs(ridge.coef)) %>% 
      ggplot(., aes(x = reorder(Predictor, -ridge.coef), y = ridge.coef,
                    fill = ridge.coef > 0)) + 
      geom_bar(stat = "identity")+ 
      labs(x = "") + 
      labs(y = "") + 
      ylim(-0.75, 0.75) + 
      theme(legend.position = "none") + #coord_flip() + 
      ggtitle((sprintf("%.1fn Estimated Coefficients (Top 20) - Ridge",
                       learn.pct[j]))) 
  } else {
    ridge.bar = ggplot(ridge.df, aes(x = reorder(Predictor, -ridge.coef),
                                     y = ridge.coef, fill = ridge.coef > 0)) + 
      geom_bar(stat = "identity")+ 
      labs(x = "") + 
      labs(y = "") + 
      ylim(-0.75, 0.75) + 
      theme(legend.position = "none") + #coord_flip() + 
      ggtitle((sprintf("%.1fn Estimated Coefficients - Ridge",learn.pct[j]))) 
    
    ridge.bar_20 = top_n(ridge.df, n = 20, abs(ridge.coef)) %>% 
      ggplot(., aes(x = reorder(Predictor, -ridge.coef), y = ridge.coef,
                    fill = ridge.coef > 0)) + 
      geom_bar(stat = "identity")+ 
      labs(x = "") + 
      labs(y = "") + 
      ylim(-0.75, 0.75) + 
      theme(legend.position = "none") + #coord_flip() + 
      ggtitle((sprintf("%.1fn Estimated Coefficients (Top 20) - Ridge",
                       learn.pct[j]))) 
  }

  #################
  # Elnet Barplots
  #################
  
  elnet.CV      =    cv.glmnet(X.os.train, y.os.train, family = "binomial",
                               alpha = 0, intercept = TRUE, standardize = FALSE, 
                               nfolds = 10, type.measure = "class")
  lam.elnet     =    exp(seq(log(max(elnet.CV$lambda)),log(0.00001), 
                             -(log(max(elnet.CV$lambda))-log(0.00001))/(m-1)))
  
  elnet.CV      =    cv.glmnet(X.os.train, y.os.train, lambda = lam.elnet, 
                               family = "binomial", alpha = 0, intercept = TRUE, 
                               standardize = FALSE, nfolds = 10,
                               type.measure = "class")
  
  elnet.fit     =    glmnet(X.os.train, y.os.train, lambda = elnet.CV$lambda.min, 
                            family = "binomial", alpha = 0,  
                            intercept = TRUE, standardize = FALSE)
  
  elnet.df = data.frame(Predictor = as.vector(row.names(elnet.fit$beta)), 
                        elnet.coef = as.vector(elnet.fit$beta))
  
  if (j == 1) {
    elnet.bar = ggplot(elnet.df, aes(x = reorder(Predictor, -elnet.coef),
                                     y = elnet.coef, fill = elnet.coef > 0)) + 
      geom_bar(stat = "identity") + 
      labs(x = "Predictors") + 
      labs(y = "") + 
      ylim(-0.75, 0.75) + 
      theme(legend.position = "none") + #coord_flip() + 
      ggtitle((sprintf("%.1fn Estimated Coefficients - Elastic-net",
                       learn.pct[j]))) 
    
    elnet.bar_20 = top_n(elnet.df, n=20, abs(elnet.coef)) %>% 
      ggplot(., aes(x = reorder(Predictor, -elnet.coef), y = elnet.coef,
                    fill = elnet.coef > 0)) + 
      geom_bar(stat = "identity")+ 
      labs(x = "Predictors") + 
      labs(y = "") + 
      ylim(-0.75, 0.75) + 
      theme(legend.position = "none") + #coord_flip() + 
      ggtitle((sprintf("%.1fn Estimated Coefficients (Top 20) - Elastic-net",
                       learn.pct[j]))) 
  } else {
    elnet.bar = ggplot(elnet.df, aes(x = reorder(Predictor, -elnet.coef),
                                     y = elnet.coef, fill = elnet.coef > 0)) + 
      geom_bar(stat = "identity")+ 
      labs(x = "Predictors") + 
      labs(y = "") + 
      ylim(-0.75, 0.75) + 
      theme(legend.position = "none") + #coord_flip() + 
      ggtitle((sprintf("%.1fn Estimated Coefficients - Elastic-net",
                       learn.pct[j]))) 
    
    elnet.bar_20 = top_n(elnet.df, n=20, abs(elnet.coef)) %>% 
      ggplot(., aes(x = reorder(Predictor, -elnet.coef), y = elnet.coef,
                    fill = elnet.coef > 0)) + 
      geom_bar(stat = "identity")+ 
      labs(x = "Predictors") + 
      labs(y = "") + 
      ylim(-0.75, 0.75) + 
      theme(legend.position = "none") + #coord_flip() + 
      ggtitle((sprintf("%.1fn Estimated Coefficients (Top 20) - Elastic-net",
                       learn.pct[j]))) 
  }  
  
  #########################
  # Random Forest Barplots
  #########################
  
  rf.fit      =   randomForest(target~., data = os.train.data, mtry = sqrt(p), 
                               importance = TRUE)
  
  rf.df = data.frame(Predictor = c(row.names(rf.fit$importance)),
                     rf.var.imp = c(rf.fit$importance[,4]))
  
  if (j == 1) {
    rf.bar = ggplot(rf.df, aes(x = reorder(Predictor, -rf.var.imp),
                               y = rf.var.imp)) + 
      geom_bar(stat = "identity", fill = "#619CFF") + #F8766D
      labs(x = "", y = "") + ylim(0, 11.5) + 
      theme(legend.position = "none") + #coord_flip() + 
      ggtitle((sprintf("%.1fn Variable Importance - Random Forest",
                       learn.pct[j]))) 
    
    rf.bar_20 = top_n(rf.df, n = 20, rf.var.imp) %>%
      ggplot(., aes(x = reorder(Predictor, -rf.var.imp), y = rf.var.imp)) + 
      geom_bar(stat = "identity", fill = "#619CFF")+ 
      labs(x = "", y = "") + ylim(0, 11.5) +
      theme(legend.position = "none") + #coord_flip() + 
      ggtitle((sprintf("%.1fn Variable Importance (Top 20) - Random Forest",
                       learn.pct[j])))  
    
  } else {
    rf.bar = ggplot(rf.df, aes(x = reorder(Predictor, -rf.var.imp),
                               y = rf.var.imp)) + 
      geom_bar(stat = "identity", fill = "#619CFF") + 
      labs(x = "", y = "") + ylim(0, 11.5) + 
      theme(legend.position = "none") + #coord_flip() + 
      ggtitle((sprintf("%.1fn Variable Importance - Random Forest",
                       learn.pct[j])))  
    
    rf.bar_20 = top_n(rf.df, n = 20, rf.var.imp) %>%
      ggplot(., aes(x = reorder(Predictor, -rf.var.imp), y = rf.var.imp)) + 
      geom_bar(stat = "identity", fill = "#619CFF") + 
      labs(x = "", y = "") + ylim(0, 11.5) +
      #coord_flip() + 
      ggtitle((sprintf("%.1fn Variable Importance (Top 20) - Random Forest",
                       learn.pct[j])))  
    
  }
  
  if (j == 1) {
    LASSO.bar1 = LASSO.bar
    ridge.bar1 = ridge.bar
    elnet.bar1 = elnet.bar
    rf.bar1    = rf.bar 
    
    LASSO.bar_20_1 = LASSO.bar_20
    ridge.bar_20_1 = ridge.bar_20
    elnet.bar_20_1 = elnet.bar_20
    rf.bar_20_1    = rf.bar_20 
  } else {
    LASSO.bar2 = LASSO.bar
    ridge.bar2 = ridge.bar
    elnet.bar2 = elnet.bar
    rf.bar2    = rf.bar 
    
    LASSO.bar_20_2 = LASSO.bar_20
    ridge.bar_20_2 = ridge.bar_20
    elnet.bar_20_2 = elnet.bar_20
    rf.bar_20_2    = rf.bar_20 
  }
  
}

# grid.arrange(LASSO.bar1, ridge.bar1, rf.bar1, 
#              LASSO.bar2, ridge.bar2, rf.bar2, ncol = 3)

# # Full 0.5n Plots
# grid.arrange(rf.bar1, LASSO.bar1, ridge.bar1, elnet.bar1, nrow = 4)

# Full 0.9n Plots
grid.arrange(rf.bar2, LASSO.bar2, ridge.bar2, elnet.bar2, nrow = 4)

# # Top 20 0.5n Plots
# grid.arrange(rf.bar_20_1, LASSO.bar_20_1, ridge.bar_20_1, elnet.bar_20_1, nrow = 4)

# Top 20 0.9n Plots
grid.arrange(rf.bar_20_2, LASSO.bar_20_2, ridge.bar_20_2, elnet.bar_20_2,
             nrow = 4)

# # All top 20 plots
# grid.arrange(LASSO.bar_20_1, ridge.bar_20_1, rf.bar_20_1, elnet.bar_20_1,
#              LASSO.bar_20_2, ridge.bar_20_2, rf.bar_20_2, elnet.bar_20_2,
#              ncol = 4)

# Checkpoint 5
save.image(file = 'Checkpoint 5 - Plots.RData')

#####################################
# Changing Variable Importance Order
#####################################

# Changing order of factor levels by specifying the order explicitly (Using RF)
rf.df$Predictor     =  factor(rf.df$Predictor,    levels = rf.df$Predictor[order(rf.df$rf.var.imp, decreasing = TRUE)])
LASSO.df$Predictor  =  factor(LASSO.df$Predictor, levels = rf.df$Predictor[order(rf.df$rf.var.imp, decreasing = TRUE)])
elnet.df$Predictor  =  factor(elnet.df$Predictor, levels = rf.df$Predictor[order(rf.df$rf.var.imp, decreasing = TRUE)])
ridge.df$Predictor  =  factor(ridge.df$Predictor, levels = rf.df$Predictor[order(rf.df$rf.var.imp, decreasing = TRUE)])

# Changing order of factor levels by specifying the order explicitly (Using LASSO)
#rf.df$Predictor     =  factor(rf.df$Predictor,    levels = LASSO.df$Predictor[order(LASSO.df$LASSO.coef, decreasing = TRUE)])
LASSO.df$Predictor  =  factor(LASSO.df$Predictor, levels = LASSO.df$Predictor[order(LASSO.df$LASSO.coef, decreasing = TRUE)])
elnet.df$Predictor  =  factor(elnet.df$Predictor, levels = LASSO.df$Predictor[order(LASSO.df$LASSO.coef, decreasing = TRUE)])
ridge.df$Predictor  =  factor(ridge.df$Predictor, levels = LASSO.df$Predictor[order(LASSO.df$LASSO.coef, decreasing = TRUE)])

# Changing order of factor levels by specifying the order explicitly (Using EN)
rf.df$Predictor     =  factor(rf.df$Predictor,    levels = elnet.df$Predictor[order(elnet.df$elnet.coef, decreasing = TRUE)])
LASSO.df$Predictor  =  factor(LASSO.df$Predictor, levels = elnet.df$Predictor[order(elnet.df$elnet.coef, decreasing = TRUE)])
elnet.df$Predictor  =  factor(elnet.df$Predictor, levels = elnet.df$Predictor[order(elnet.df$elnet.coef, decreasing = TRUE)])
ridge.df$Predictor  =  factor(ridge.df$Predictor, levels = elnet.df$Predictor[order(elnet.df$elnet.coef, decreasing = TRUE)])

# Re-executing plots

########
# Ridge
########

rf.bar = ggplot(rf.df, aes(x = Predictor,
                           y = rf.var.imp)) + 
  geom_bar(stat = "identity", fill = "#619CFF") + 
  labs(x = "", y = "") + ylim(0, 11.5) + 
  theme(legend.position = "none") + #coord_flip() + 
  ggtitle((sprintf("%.1fn Variable Importance - Random Forest",
                   learn.pct[j])))  

rf.bar_20 = top_n(rf.df, n = 20, rf.var.imp) %>%
  ggplot(., aes(x = Predictor, y = rf.var.imp)) + 
  geom_bar(stat = "identity", fill = "#619CFF") + 
  labs(x = "", y = "") + ylim(0, 11.5) +
  #coord_flip() + 
  ggtitle((sprintf("%.1fn Variable Importance (Top 20) - Random Forest",
                   learn.pct[j])))

########
# Elnet
########

elnet.bar = ggplot(elnet.df, aes(x = Predictor,
                                 y = elnet.coef, fill = elnet.coef > 0)) + 
  geom_bar(stat = "identity")+ 
  labs(x = "Predictors") + 
  labs(y = "") + 
  ylim(-0.75, 0.75) + 
  theme(legend.position = "none") + #coord_flip() + 
  ggtitle((sprintf("%.1fn Estimated Coefficients - Elastic-net",learn.pct[j]))) 

elnet.bar_20 = top_n(elnet.df, n=20, abs(elnet.coef)) %>% 
  ggplot(., aes(x = Predictor, y = elnet.coef,
                fill = elnet.coef > 0)) + 
  geom_bar(stat = "identity")+ 
  labs(x = "Predictors") + 
  labs(y = "") + 
  ylim(-0.75, 0.75) + 
  theme(legend.position = "none") + #coord_flip() + 
  ggtitle((sprintf("%.1fn Estimated Coefficients (Top 20) - Elastic-net",
                   learn.pct[j])))

########
# Ridge
########

ridge.bar = ggplot(ridge.df, aes(x = Predictor,
                                 y = ridge.coef, fill = ridge.coef > 0)) + 
  geom_bar(stat = "identity")+ 
  labs(x = "") + 
  labs(y = "") + 
  ylim(-0.75, 0.75) + 
  theme(legend.position = "none") + #coord_flip() + 
  ggtitle((sprintf("%.1fn Estimated Coefficients - Ridge",learn.pct[j]))) 

ridge.bar_20 = top_n(ridge.df, n = 20, abs(ridge.coef)) %>% 
  ggplot(., aes(x = Predictor, y = ridge.coef,
                fill = ridge.coef > 0)) + 
  geom_bar(stat = "identity")+ 
  labs(x = "") + 
  labs(y = "") + 
  ylim(-0.75, 0.75) + 
  theme(legend.position = "none") + #coord_flip() + 
  ggtitle((sprintf("%.1fn Estimated Coefficients (Top 20) - Ridge",
                   learn.pct[j])))

########
# LASSO
########

LASSO.bar = ggplot(LASSO.df, aes(x = Predictor,
                                 y = LASSO.coef, fill = LASSO.coef > 0)) + 
  geom_bar(stat = "identity") + 
  labs(x = "") + 
  labs(y = "Values") + 
  ylim(-0.75, 0.75) + 
  theme(legend.position = "none") + #coord_flip() + 
  ggtitle((sprintf("%.1fn Estimated Coefficients - LASSO",learn.pct[j]))) 

LASSO.bar_20 = top_n(LASSO.df, n = 20, abs(LASSO.coef)) %>% 
  ggplot(., aes(x = Predictor, y = LASSO.coef,
                fill = LASSO.coef > 0)) + 
  geom_bar(stat = "identity") + 
  labs(x = "") + 
  labs(y = "Values") + 
  ylim(-0.75, 0.75) + 
  theme(legend.position = "none") + #coord_flip() + 
  ggtitle((sprintf("%.1fn Estimate Coefficients (Top 20) - LASSO",
                   learn.pct[j]))) 

# Full 0.9n Plots
grid.arrange(rf.bar, LASSO.bar, ridge.bar, elnet.bar, nrow = 4)

# Top 20 0.9n Plots
grid.arrange(rf.bar_20_2, LASSO.bar_20_2, ridge.bar_20_2, elnet.bar_20_2,
             nrow = 4)

# Checkpoint 6
#save.image(file = 'Checkpoint 6 - Complete.RData')



###############################################################################
##################################### END #####################################
###############################################################################