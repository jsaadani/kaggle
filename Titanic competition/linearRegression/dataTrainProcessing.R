rm(list = ls())
source("./linearRegression/linearRegAgeParams.R")
data <- read.csv("./data/train.csv")
names(data) <- tolower(names(data))

#====replace NA with linear regression values=========================
replaceData <- data[is.na(data$age),]
replaceData <- replaceData[,c("pclass","sex","sibsp","parch","fare","embarked","age")]
replaceData$sex <- as.numeric(replaceData$sex)-1
replaceData$embarked <- as.numeric(replaceData$embarked)-1
Xtrain <- replaceData[,c("pclass","sibsp","parch","fare","embarked")]

#====map polynomial features====
source("./functions/polyFeatures.R")
Xtrain<- polyFeatures(Xtrain,8)

#=====normalize test set====
source("./functions/normVal.R")
Xtrain <- normVal(Xtrain,mu,sigma)

#====add sex variable
Xtrain <- cbind(Xtrain,replaceData[,c("sex")])

#====put data into matrix====
Xtrain <- as.matrix(Xtrain)
#====Add intercept term====
Xtrain <- cbind(matrix(1,nrow(Xtrain)),Xtrain)
data$age[is.na(data$age)] <- Xtrain%*%theta
data$fare[is.na(data$fare)] <- mean(data$fare, na.rm=TRUE)

#====export====
dump("data",file="./linearRegression/dataTrainAgeReplaced.R")