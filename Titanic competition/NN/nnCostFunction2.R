nnCostFunction <- function(nn_params,
                           input_layer_size,
                           hidden_layer_size_1,
                           hidden_layer_size_2,
                           num_labels,
                           X,y,lambda){
  
  #load functions
  source("sigmoid.R")
  
  #test
  #nn_params=initial_nn_params
  
  #reshape nn_params into Theta1,Theta2, Theta3
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
  J=0
  
  #Cost Function
  #Layer 2
  X <- cbind(matrix(1,m),X)#add bias unit
  z2 <-Theta1%*%t(X)
  a2 <- sigmoid(z2)
  
  #Layer 3
  a2 <- rbind(matrix(1,1,m),a2)#add bias unit
  z3 <- Theta2%*%a2
  a3 <- sigmoid(z3) #dimension: num_labels*m
  
  #Layer 4
  a3 <- rbind(matrix(1,1,m),a3)#add bias unit
  z4 <- Theta3%*%a3
  a4 <- sigmoid(z4)
  h <- matrix(a4, ncol=1)
  
  #   for(i in 1:m){
#     for(k in 1:num_labels){
#       J=J-(y[i]==k)*log(h[k,i])-(1-(y[i]==k))*log(1-h[k,i])
#     }
#   }
#   J=J/m
  
  J=-(1/m)*sum(y*log(h)+(1-y)*log(1-h))
  
  #Regularized cost function
  Theta1_reg <- Theta1[,2:ncol(Theta1)]
  Theta2_reg <- Theta2[,2:ncol(Theta2)]
  Theta3_reg <- Theta3[,2:ncol(Theta3)]
  
  cost_Theta1=sum(Theta1_reg^2)
  cost_Theta2=sum(Theta2_reg^2)
  cost_Theta3=sum(Theta3_reg^2)
  
  J=J+(lambda/(2*m))*(cost_Theta1+cost_Theta2+cost_Theta3)
  
  #backprop
#   Theta1_grad <- matrix(0,nrow(Theta1),ncol(Theta1))
#   Theta2_grad <- matrix(0,nrow(Theta2),ncol(Theta2))
#   for(t in 1:m){
#     
#     a1 <- t(X[t,])
#     z2 <- Theta1%*%a1
#     a2 <- sigmoid(z2)
#     
#     a2 <- rbind(1,a2)#add bias unit (verify dim(a2))
#     z3 <- Theta2%*%a2
#     a3 <- sigmoid(z3)
#     
#     #error layer3
#     delta3 <- matrix(0,nrow(a3),ncol(a3))
#     for(k in 1:num_labels){
#       delta3[k] <- a3[k]-(y[t]==k)
#     }
#     
#     #error layer 2
#     delta2 <- matrix(0,nrow(a2),ncol(a2))
#     Theta2_back <- Theta2[,2:ncol(Theta2)]
#     delta2 <- t(Theta2_back)%*%delta3*sigmoidGradient(z2)
#     
#     Theta1_grad <- Theta1_grad+delta2%*%t(a1)
#     Theta2_grad <- Theta2_grad+delta3%*%t(a2)
#     
# 
#  
#   }
#   
#   #regularization
#   reg1 <- lambda*Theta1
#   reg1 <- cbind(matrix(0,nrow(reg1)),reg1[,2:ncol(reg1)])
#   
#   reg2 <- lambda*Theta2
#   reg2 <- cbind(matrix(0,nrow(reg2)),reg1[,2:ncol(reg2)])
#   
#   Theta1_grad <- Theta1_grad+reg1
#   Theta2_grad <- Theta2_grad+reg2
#   
#   #divide by m
#   Theta1_grad <- Theta1_grad/m
#   Theta2_grad <- Theta2_grad/m
#   
#   #unroll gradients
#   grad <- matrix(c(Theta1_grad,Theta2_grad),ncol=1)
  
return(J)
  
}