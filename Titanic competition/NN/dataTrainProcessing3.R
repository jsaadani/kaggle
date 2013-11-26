#Closing all the plots and resetting all the variables in the workspace
#graphics.off()
rm(list = ls())

#load "data"
source("../linearRegression/dataTrainAgeReplaced.R")
#load "dataTest"
source("../linearRegression/dataTestAgeReplaced.R")

#====split train set and CV set====
#select columns
trainData <- data[,c("survived","pclass","sex","sibsp","parch","fare","embarked","age","cabin")]
testData <- dataTest[,c("pclass","sex","sibsp","parch","fare","embarked","age","cabin")]
trainData$sex <- as.numeric(trainData$sex)-1
trainData$embarked <- as.numeric(trainData$embarked)-1
testData$sex <- as.numeric(testData$sex)-1
testData$embarked <- as.numeric(testData$embarked)-1
##process "cabin"
#transform into numeric variable
trainData$cabin <- as.numeric(trainData$cabin)
testData$cabin <- as.numeric(testData$cabin)
#transform into dummy variable for trainData
for(i in 1:nrow(trainData)){
  if(trainData$cabin[i]==1){
    trainData$cabin[i] <- 0
  }
  else{
    trainData$cabin[i] <- 1
  }
}
#transform into dummy variable for testData
for(i in 1:nrow(testData)){
  if(testData$cabin[i]==1){
    testData$cabin[i] <- 0
  }
  else{
    testData$cabin[i] <- 1
  }
}


#split
set.seed(2)
rand <- sample(nrow(trainData))
randData <- trainData[rand,]
c <- floor(0.8*nrow(randData))#determines where to cut
trainSet <- randData[1:c,]#80% train set
cvSet <- randData[(c+1):nrow(randData),]#20% CV set
testSet <- testData

#====define X,y,Xval,yval,Xtest====
#Ajouter sex après map features
X <- trainSet[,c("pclass","sibsp","parch","fare","embarked","age")]
y <- trainSet[,c("survived")]
Xval <- cvSet[,c("pclass","sibsp","parch","fare","embarked","age")]
yval <- cvSet[,c("survived")]
Xtest <- testSet[c("pclass","sibsp","parch","fare","embarked","age")]

# #====map polynomial features====
# source("./functions/polyFeatures.R")
# X <- polyFeatures(X,6)
# Xval <- polyFeatures(Xval,6)
# Xtest <- polyFeatures(Xtest,6)

#====Normalize====
#normalize train set
source("../functions/featureNormalize.R")
normalizeTrain <- featureNormalize(X)
X <- normalizeTrain$norm
mu <- normalizeTrain$mu
sigma <- normalizeTrain$sigma
#normalize validation set
source("../functions/normVal.R")
Xval <- normVal(Xval,mu,sigma)
Xtest <- normVal(Xtest,mu,sigma)

#====add sex variable
X <- cbind(X,trainSet[,c("sex","cabin")])
Xval <- cbind(Xval,cvSet[,c("sex","cabin")])
Xtest <- cbind(Xtest,testSet[,c("sex","cabin")])

#====put data into matrix====
X <- as.matrix(X)
y <- as.matrix(y)
Xval <- as.matrix(Xval)
yval <- as.matrix(yval)
Xtest <- as.matrix(Xtest)

# dump(c("Xval","yval"),file="cvSet.R")
#dump(c("X","y","Xval","yval","Xtest"),file="./data/dataProcessedLinearAge/trainValidationTestSet.R")
save(X,y,Xval,yval,Xtest,file="NN_data3.rda")