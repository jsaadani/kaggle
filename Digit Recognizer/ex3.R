#resetting all the variables in the workspace
rm(list=ls())

#load functions
source("oneVsAll.R")
source("predictOneVsAll.R")

#load training data "X" and "y"
source("ex3data1.R")

#set some useful values
num_labels=10
lambda=0.1

#====learn Theta====
cat("Training One-vs-All Logistic Regression...\n")
all_theta <- oneVsAll(X,y,num_labels,lambda)

#Pause
cat("Program paused. Press enter to continue...")
readline()

#====Predict One Vs All=====
pred <- predictOneVsAll(all_theta,X)

cat("Training set accuracy: ",mean(as.numeric(pred==y)),"\n")




