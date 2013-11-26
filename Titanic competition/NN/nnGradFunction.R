nnGradFunction <- function(nn_params,
                             input_layer_size,
                             hidden_layer_size,
                             num_labels,
                             X,y,lambda){
  
  #test
  #nn_params=initial_nn_params
  
  
  #load functions
  source("sigmoid.R")
  source("sigmoidGradient.R")
  
  
  #reshape nn_params into Theta1 and Theta2
  Theta1 <- nn_params[1:(hidden_layer_size*(input_layer_size+1))]
  dim(Theta1) <- c(hidden_layer_size,(input_layer_size+1))
  Theta2 <- nn_params[(hidden_layer_size*(input_layer_size+1)+1):length(nn_params)]
  dim(Theta2) <- c(num_labels,(hidden_layer_size+1))
  
  #setup some useful values
  m=nrow(X)
  
  #variables to return
  Theta1_grad <- matrix(0,nrow(Theta1),ncol(Theta1))
  Theta2_grad <- matrix(0,nrow(Theta2),ncol(Theta2))
  
  #backprop
  X <- cbind(matrix(1,m),X)#add bias unit
  for(t in 1:m){
    #test
    #t=1
    a1 <- matrix((X[t,]),ncol=1)
    z2 <- Theta1%*%a1
    a2 <- sigmoid(z2)
    
    a2 <- rbind(1,a2)#add bias unit (verify dim(a2))
    z3 <- Theta2%*%a2
    a3 <- sigmoid(z3)
    
    #error layer3
    delta3 <- matrix(0,nrow(a3),ncol(a3))
#     for(k in 1:num_labels){
#       delta3[k] <- a3[k]-(y[t]==k)
#       }
      delta3=a3-y[t]
    
    #error layer 2
    delta2 <- matrix(0,nrow(a2),ncol(a2))
    Theta2_back <- matrix(Theta2[,2:ncol(Theta2)],ncol=ncol(Theta2)-1)
    delta2 <- t(Theta2_back)%*%delta3*sigmoidGradient(z2)
    
    Theta1_grad <- Theta1_grad+delta2%*%t(a1)
    Theta2_grad <- Theta2_grad+delta3%*%t(a2)
  }
  
  #regularization
  reg1 <- lambda*Theta1
  reg1 <- cbind(matrix(0,nrow(reg1)),reg1[,2:ncol(reg1)])
  
  reg2 <- lambda*Theta2
  reg2 <- cbind(matrix(0,nrow(reg2)),matrix(reg2[,2:ncol(reg2)],ncol=ncol(reg2)-1))
  #slice <- matrix(reg2[,2:ncol(reg2)],ncol=ncol(reg2)-1)
  
  Theta1_grad <- Theta1_grad+reg1
  Theta2_grad <- Theta2_grad+reg2
  
  #divide by m
  Theta1_grad <- Theta1_grad/m
  Theta2_grad <- Theta2_grad/m
  
  #unroll gradients
  grad <- matrix(c(Theta1_grad,Theta2_grad),ncol=1)
  
  return(grad)
  
}

    