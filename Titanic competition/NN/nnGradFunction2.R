nnGradFunction <- function(nn_params,
                             input_layer_size,
                             hidden_layer_size_1,
                            hidden_layer_size_2,
                             num_labels,
                             X,y,lambda){
  
  #test
  #nn_params=initial_nn_params
  
  
  #load functions
  source("sigmoid.R")
  source("sigmoidGradient.R")
  
  
  #reshape nn_params into Theta1,Theta2,Theta3
  len_Theta1=hidden_layer_size_1*(input_layer_size+1)
  len_Theta2=hidden_layer_size_2*(hidden_layer_size_1+1)
  len_Theta3=num_labels*(hidden_layer_size_2+1)
  
  Theta1 <- nn_params[1:len_Theta1]
  dim(Theta1) <- c(hidden_layer_size_1,input_layer_size+1)
  Theta2 <- nn_params[(1+len_Theta1):(len_Theta1+len_Theta2)]
  dim(Theta2) <- c(hidden_layer_size_2,(hidden_layer_size_1+1))
  Theta3 <- nn_params[(1+len_Theta1+len_Theta2):length(nn_params)]
  dim(Theta3) <- c(num_labels,(hidden_layer_size_2+1))
  
  #setup some useful values
  m=nrow(X)
  
  #variables to return
  Theta1_grad <- matrix(0,nrow(Theta1),ncol(Theta1))
  Theta2_grad <- matrix(0,nrow(Theta2),ncol(Theta2))
  Theta3_grad <- matrix(0,nrow(Theta3),ncol(Theta3))
  
  #grad
  X <- cbind(matrix(1,m,1),X)#add bias unit
  for(t in 1:m){
    #forward prop
    #Layer 2
    a1 <- matrix((X[t,]),ncol=1)
    z2 <- Theta1%*%a1
    a2 <- sigmoid(z2)
    #layer3
    a2 <- rbind(1,a2)#add bias unit (verify dim(a2))
    z3 <- Theta2%*%a2
    a3 <- sigmoid(z3)
    #layer4
    a3 <- rbind(1,a3)#add bias unit
    z4 <- Theta3%*%a3
    a4 <- sigmoid(z4)
    
    #back prop
    
    #error layer4
    delta4 <- matrix(0,nrow(a4),ncol(a4))
    delta4 <- a4-y[t]
    
    #error layer 3
    delta3 <- matrix(0,nrow(a3),ncol(a3))
    Theta3_back <- matrix(Theta3[,2:ncol(Theta3)],ncol=ncol(Theta3)-1)
    delta3 <- t(Theta3_back)%*%delta4*sigmoidGradient(z3)
    
    #error layer 2
    delta2 <- matrix(0,nrow(a2),ncol(a2))
    Theta2_back <- matrix(Theta2[,2:ncol(Theta2)],ncol=ncol(Theta2)-1)
    delta2 <- t(Theta2_back)%*%delta3*sigmoidGradient(z2)
    
    #accumulate
    Theta1_grad <- Theta1_grad+delta2%*%t(a1)
    Theta2_grad <- Theta2_grad+delta3%*%t(a2)
    Theta3_grad <- Theta3_grad+delta4%*%t(a3)
  }
  
  #regularization
  reg1 <- lambda*Theta1
  reg1 <- cbind(matrix(0,nrow(reg1)),reg1[,2:ncol(reg1)])
  
  reg2 <- lambda*Theta2
  reg2 <- cbind(matrix(0,nrow(reg2)),matrix(reg2[,2:ncol(reg2)],ncol=ncol(reg2)-1))
  
  reg3 <- lambda*Theta3
  reg3 <- cbind(matrix(0,nrow(reg3)),matrix(reg3[,2:ncol(reg3)],ncol=ncol(reg3)-1))
  #slice <- matrix(reg2[,2:ncol(reg2)],ncol=ncol(reg2)-1)
  
  Theta1_grad <- Theta1_grad+reg1
  Theta2_grad <- Theta2_grad+reg2
  Theta3_grad <- Theta3_grad+reg3
  
  #divide by m
  Theta1_grad <- Theta1_grad/m
  Theta2_grad <- Theta2_grad/m
  Theta3_grad <- Theta3_grad/m
  
  #unroll gradients
  grad <- matrix(c(Theta1_grad,Theta2_grad,Theta3_grad),ncol=1)
  
  return(grad)
  
}

    