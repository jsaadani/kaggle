#resetting all the variables in the workspace
rm(list=ls())

#====load data====
initial <- read.csv("./data/train.csv", nrows=100)
classes <- sapply(initial,class)

dataTrain <- read.csv("./data/train.csv",
                      comment.char="",
                      colClasses=classes,
                      nrows=42000)

#====split into trainSet and cvSet====
set.seed(123)
trainIndicator <- rbinom(42000,size=1,prob=0.8)
trainSet <- dataTrain[trainIndicator==1,]
cvSet <- dataTrain[trainIndicator==0,]

#====defining X,Xval,y,yval====
y <- trainSet$label
X <- trainSet[,-1]
yval <- cvSet$label
Xval <- cvSet[,-1]

#replace 0 by 10 in y and yval
#incrementation starts at 1 in R (vs.0)
y[y==0] <- 10
yval[yval==0] <- 10

#====normalize X,y====
source("./functions/featureNormalize.R")
cat("Normalizing data...\n")
normalizeTrain <- featureNormalize(X)
X <- normalizeTrain$norm
mu <- normalizeTrain$mu
sigma <- normalizeTrain$sigma

#====normalize Xval,yval====
source("./functions/normVal.R")
Xval <- normVal(Xval,mu,sigma)

#export
save(X,y,Xval,yval,file="./data/trainSet.rda")
save(mu,sigma,file="normalizeParams.rda")
