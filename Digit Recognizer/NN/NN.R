#remove all variables from workspace
rm(list=ls())

#load functions
source("sigmoid.R")
source("predict.R")
source("nnCostFunction.R")
source("nnGradFunction.R")

#====load X,y,Xval,yval====
cat("loading training data X and y \n")
load("trainSet.rda")
X <- as.matrix(X)
y <- as.matrix(y)
Xval <- as.matrix(Xval)
yval <- as.matrix(yval)

#====NN parameters====
#layers
input_layer_size=784
hidden_layer_size=50
num_labels=10

#Initializing parameters
source("randInitializeWeights.R")
initial_Theta1 <-randInitializeWeights(input_layer_size,hidden_layer_size)
initial_Theta2 <-randInitializeWeights(hidden_layer_size,num_labels)

#unroll parameters
initial_nn_params <- matrix(c(initial_Theta1,initial_Theta2),ncol=1)

#====initial cost====
lambda=0
debug_J=nnCostFunction(initial_nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,lambda)
# cat("Cost at parameters (with regularization): ",debug_J,"\n")

#====Training NN====
cat("Training neural networks...\n")
lambda=0
result <- optim(par=initial_nn_params,fn=nnCostFunction,gr=nnGradFunction,
                input_layer_size=input_layer_size,
                hidden_layer_size=hidden_layer_size,
                num_labels=num_labels,
                X=X,y=y,lambda=lambda,
                method="BFGS")
nn_params <- result$par

#reshape nn_params into Theta1 and Theta2
Theta1 <- nn_params[1:(hidden_layer_size*(input_layer_size+1))]
dim(Theta1) <- c(hidden_layer_size,(input_layer_size+1))
Theta2 <- nn_params[(hidden_layer_size*(input_layer_size+1)+1):length(nn_params)]
dim(Theta2) <- c(num_labels,(hidden_layer_size+1))

#====predict====
pred <- predict(X,Theta1,Theta2)
cat("Training set Accuracy: ",mean(as.numeric(pred==y))*100,"\n")
cat("Train error: ",nnCostFunction(initial_nn_params,input_layer_size,hidden_layer_size,num_labels,
                                   X,y,0),"\n")

pred <- predict(Xval,Theta1,Theta2)
cat("Validation set Accuracy: ",mean(as.numeric(pred==yval))*100,"\n")
cat("CV error: ",nnCostFunction(initial_nn_params,input_layer_size,hidden_layer_size,num_labels,
                                Xval,yval,0),"\n")

#test set
#load("./data/testSet.rda")
# pred <- as.numeric(predict(Xtest,Theta1,Theta2))
# save(pred,file="predictions_NN_2.rda")