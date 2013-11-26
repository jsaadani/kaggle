source("linearRegAgeParams.R")
data <- read.csv("./data/test.csv")
names(data) <- tolower(names(data))

#====replace NA with linear regression values=========================
replaceData <- data[is.na(data$age),]
replaceData <- replaceData[,c("pclass","sex","sibsp","parch","fare","embarked","age")]
replaceData$sex <- as.numeric(replaceData$sex)-1
replaceData$embarked <- as.numeric(replaceData$embarked)-1
Xtest <- replaceData[,c("pclass","sex","sibsp","parch","fare","embarked")]

#====map polynomial features====
# source("./functions/polyFeatures.R")
# Xtest<- polyFeatures(Xtest,2)
#=====normalize test set====
source("./functions/normVal.R")
Xtest <- normVal(Xtest,mu,sigma)
#====put data into matrix====
Xtest <- as.matrix(Xtest)
#====Add intercept term====
Xtest <- cbind(matrix(1,nrow(Xtest)),Xtest)
data$age[is.na(data$age)] <- Xtest%*%theta
data$fare[is.na(data$fare)] <- mean(data$fare, na.rm=TRUE)
dataTest <- data

#====export====
dump("dataTest",file="./data/dataProcessedLinearAge/dataTestAgeReplaced.R")