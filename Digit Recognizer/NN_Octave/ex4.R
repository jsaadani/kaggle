#remove all variables from workspace
rm(list=ls())

#load functions
source("sigmoid.R")
source("predict.R")
source("nnCostFunction.R")
source("nnGradFunction.R")

#====load training data "X" and "y"====
cat("loading training data X and y \n")
load("ex4data1.rda")
X <- as.matrix(X)
y <- as.matrix(y)

#====load the weights into Theta1 and Theta2====
cat("loading saved NN parameters Theta1 and Theta2...\n")
load("ex4weights.rda")
Theta1 <- as.matrix(Theta1)
Theta2 <- as.matrix(Theta2)

#Unroll parameters
nn_params <- matrix(c(Theta1,Theta2),ncol=1)

#====Compute Cost====
source("nnCostFunction.R")
input_layer_size=400
hidden_layer_size=25
num_labels=10
cat("Feedforward using NN...\n")
lambda=0
J=nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,lambda)
cat("Cost at parameters: ",J,"\n")

#pause
cat("Program paused, press a key to continue...\n")
readline()
#checking cost function with regularization
lambda=1
J=nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,lambda)
cat("Cost at parameters: ",J,"\n")

#Initializing parameters
source("randInitializeWeights.R")
initial_Theta1 <-randInitializeWeights(input_layer_size,hidden_layer_size)
initial_Theta2 <-randInitializeWeights(hidden_layer_size,num_labels)

#unroll parameters
initial_nn_params <- matrix(c(initial_Theta1,initial_Theta2),ncol=1)

#====check regularization====
lambda=3
debug_J=nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,lambda)
cat("Cost at parameters (with regularization): ",debug_J,"\n")

#====Training NN====
cat("Training neural networks...\n")
lambda=1
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

cat("Program pause. Press enter to continue...\n")
readline()

#predict
pred <- predict(X, Theta1,Theta2)
cat("Training set Accuracy: ",mean(as.numeric(pred==y)),"\n")

