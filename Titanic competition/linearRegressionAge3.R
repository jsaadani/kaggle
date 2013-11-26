#Closing all the plots and resetting all the variables in the workspace
#graphics.off()
rm(list = ls())

#====Load functions====
source("./functions/linearRegCostFunction.R")
source("./functions/linearRegGradFunction.R")
source("./functions/polyFeatures.R")

#====reading data====
data <- read.csv("./data/train.csv")
dataTest <- read.csv("./data/test.csv")

#====missing values====
good<- complete.cases(data)
cData <- data[good,]
trainData <- cData[,c("pclass","sex","sibsp","parch","fare","embarked","age")]
trainData$sex <- as.numeric(trainData$sex)-1
trainData$embarked <- as.numeric(trainData$embarked)-1

#====split train set and CV set====
set.seed(2)
rand <- sample(nrow(trainData))
randData <- trainData[rand,]
c <- floor(0.7*nrow(randData))#determines where to cut
trainSet <- randData[1:c,]#70% train set
cvSet <- randData[(c+1):nrow(randData),]#30% CV set

#====define X,y,Xval,yval====
X <- trainSet[,c("pclass","sibsp","parch","fare","embarked")]
y <- trainSet[,c("age")]
Xval <- cvSet[,c("pclass","sibsp","parch","fare","embarked")]
yval <- cvSet[,c("age")]

#====map polynomial features====
X <- polyFeatures(X,8)
Xval <- polyFeatures(Xval,8)

#====Normalize====
#=normalize train set
source("./functions/featureNormalize.R")
normalizeTrain <- featureNormalize(X)
X <- normalizeTrain$norm
mu <- normalizeTrain$mu
sigma <- normalizeTrain$sigma
#=normalize validation set
source("./functions/normVal.R")
Xval <- normVal(Xval,mu,sigma)

#====add sex variable
X <- cbind(X,trainSet[,c("sex")])
Xval <- cbind(Xval,cvSet[,c("sex")])

#====put data into matrix====
X <- as.matrix(X)
y <- as.matrix(y)
Xval <- as.matrix(Xval)
yval <- as.matrix(yval)

#====setup  some useful values====
m=nrow(X)
n=ncol(X)

#====Add intercept term====
X <- cbind(matrix(1,nrow(X)),X)
Xval <- cbind(matrix(1,nrow(Xval)),Xval)

#====Initialize fitting parameters====
#initialize theta and lambda
initial_theta <- matrix(0,n+1,1)
lambda=30

#====initial cost====
cat("initial cost :",linearRegCostFunction(initial_theta,X,y,lambda),"\n")
#Pause
cat("Program paused. Press enter to continue...")
readline()

#====learn theta====
linearRegCostFunction(initial_theta,X,y,lambda)
result <- optim(par=initial_theta,fn=linearRegCostFunction,gr=linearRegGradFunction,X=X,y=y,lambda=lambda,
                method="BFGS")
theta <- result$par
cost <- result$value
#====Training error====
cat('Train error: ', linearRegCostFunction(theta,X,y,0),"\n")
#====Validation error====
cat('Validation error,lambda=0: ', linearRegCostFunction(theta,Xval,yval,0),"\n")

# #Pause
# cat("Program paused. Press enter to continue...")
# readline()

# #====Learning curve====
# cat("Plotting learning curve...")
# source("./functions/learningCurve.R")
# source("./functions/trainLinearReg.R")
# lambda=0
# error_train<- learningCurve(X,y,Xval,yval,lambda)$error_train
# error_val <- learningCurve(X,y,Xval,yval,lambda)$error_val
# 
# #====plot learning curve====
# plot(2:m,error_train,
#      type="l",
#      lwd=1,
#      col="blue",
#      main="Learning curve for linear regression",
#      ylab="Error",
#      xlab="Number of training examples",
#      cex.lab=0.9,
#      xlim=c(0,m),
#      ylim=c(0,200))
# 
# lines(2:m,error_val,col="green",lwd=1)
# legend(400,150,c("Train","Cross Validation"),col=c("blue","green"), cex=0.5, xjust=1, lwd=1)

#  cat("Program paused. Press enter to continue...")
#  readline()

# #====validation for selecting lambda====
# cat("plotting validation curve...\n")
# source("./functions/validationCurve.R")
# vc <- validationCurve(X,y,Xval,yval)
# error_train <- vc$error_train
# error_val <- vc$error_val
# lambda_vec <- vc$lambda_vec
# 
# #====plot validaion curve====
# plot(lambda_vec,error_train,
#      type="l",
#      lwd=1,
#      col="blue",
#      main="validation curve",
#      ylab="Error",
#      xlab="lambda",
#      cex.lab=0.9,
#      xlim=c(0,max(lambda_vec)),
#      ylim=c(60,95))
# lines(lambda_vec,error_val,col="green",lwd=1)
# 
# cat("Lambda\t\tTrain error\t\tValidation error\n")
# for(i in 1:length(lambda_vec)){
#   cat(lambda_vec[i],"\t\t")
#   cat(error_train[i],"\t\t")
#   cat(error_val[i],"\n")
# }

# legend(400,150,c("Train","Cross Validation"),col=c("blue","green"), cex=0.5, xjust=1, lwd=1)

# #====replace NA with linear regression values=========================
# replaceData <- data[is.na(data$age),]
# replaceData <- replaceData[,c("survived","pclass","sex","sibsp","parch","fare","embarked","age")]
# replaceData$sex <- as.numeric(replaceData$sex)-1
# replaceData$embarked <- as.numeric(replaceData$embarked)
# Xtest <- replaceData[,c("survived","pclass","sex","sibsp","parch","fare","embarked")]
# #ytest <- replaceData[,c("age")]
# 
# #====map polynomial features====
# Xtest<- polyFeatures(Xtest,2)
# #=====normalize test set====
# source("./functions/normVal.R")
# Xtest <- normVal(Xtest,mu,sigma)
# #====put data into matrix====
# Xtest <- as.matrix(Xtest)
# # #====setup  some useful values====
# # m=nrow(Xtest)
# # n=ncol(Xtest)
# #====Add intercept term====
# Xtest <- cbind(matrix(1,nrow(Xtest)),Xtest)
# sum(Xtest%*%theta<0)
# data$age[is.na(data$age)] <- Xtest%*%theta
# sum(is.na(data$age))
# head(data$age)
# dump("data",file="trainAgeReplaced.R")

dump(c("theta","mu","sigma"),file="./linearRegression/linearRegAgeParams.R")