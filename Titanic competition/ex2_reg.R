#Closing all the plots and resetting all the variables in the workspace
graphics.off()
rm(list = ls())

#Loading files filled in manually
source("sigmoid.R")
source("costFunctionReg.R")
source("gradFunctionReg.R")

#Reading data
data2 <-read.table(file="ex2data2.txt",header=FALSE, sep=",")
names(data2) <- c("test1","test2","accepted")
X <- as.matrix(data2[,1:2])
y <- as.matrix(data2[,3])

#Plotting data
library(lattice)
xyplot( test2 ~ test1 , data=data2, groups=factor(y),
        panel=function(...){
          panel.xyplot(...)
        })

#Feature mapping
source("mapFeature.R")
X <- mapFeature(X[,1],X[,2])

#====Compute Cost====
#setup  some useful values
m=nrow(X)
n=ncol(X)

#Initialize fitting parameters
initial_theta=matrix(0,n,1)
lambda=1
costFunctionReg(initial_theta,X,y,lambda)


result <- optim(par=initial_theta,fn=costFunctionReg,gr=gradFunctionReg,X=X,y=y,lambda=1,
                method="BFGS")
theta <- result$par
cost <- result$value


#====Train accuracy====
# source("predict.R")
# p <- predict(theta,X)
trainAccuracy <- mean(as.numeric(p==y))*100
sprintf('Train accuracy: %f', trainAccuracy)

p <- sigmoid(X%*%theta)>=0.5
trainAccuracy <- mean(p==y)*100
sprintf('Train accuracy: %f', trainAccuracy)


