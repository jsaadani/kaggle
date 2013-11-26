#resetting all the variables in the workspace
rm(list=ls())

#load data
initial <- read.csv("./data/test.csv", nrows=100)
classes <- sapply(initial,class)
dataTest <- read.csv("./data/test.csv",
                      comment.char="",
                      colClasses=classes,
                      nrows=28000)

#defining Xtest
Xtest <- dataTest

#====normalize X====
source("./functions/normVal.R")
load("normalizeParams.rda")
cat("Normalizing test Set...\n")
Xtest <- normVal(Xtest,mu,sigma)

#export
#dump("Xtest",file="testSet.R")
save(Xtest,file="./data/testSet.rda")