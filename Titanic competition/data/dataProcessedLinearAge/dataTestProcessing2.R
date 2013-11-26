#Closing all the plots and resetting all the variables in the workspace
graphics.off()
rm(list = ls())

#load "data"
source("./data/dataProcessedLinearAge/dataTestAgeReplaced.R")

#====split train set and CV set====
#select columns
testData <- data[,c("survived","pclass","sex","sibsp","parch","fare","embarked","age")]
testData$sex <- as.numeric(testData$sex)-1
testData$embarked <- as.numeric(testData$embarked)-1

#====define X,y,Xval,yval====
Xtest <- trainSet[,c("pclass","sex","sibsp","parch","fare","embarked","age")]

#====Normalize====
#normalize train set
source("./functions/featureNormalize.R")
normalizeTrain <- featureNormalize(X)
X <- normalizeTrain$norm
mu <- normalizeTrain$mu
sigma <- normalizeTrain$sigma
#normalize validation set
source("./functions/normVal.R")
Xval <- normVal(Xval,mu,sigma)

#====put data into matrix====
X <- as.matrix(X)
y <- as.matrix(y)
Xval <- as.matrix(Xval)
yval <- as.matrix(yval)

# dump(c("Xval","yval"),file="cvSet.R")
dump(c("X","y","Xval","yval"),file="./data/dataProcessedLinearAge/trainSet.R")