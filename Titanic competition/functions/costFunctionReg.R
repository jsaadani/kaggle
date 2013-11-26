costFunctionReg <- function(theta,X,y,lambda){
  
  #test
  #theta=initial_theta
  
  #initialize some useful values
  m=nrow(X)
  #value to return
  #J=0
  
  #cost function
  predictions=sigmoid(X%*%theta)
  theta_reg=theta[2:length(theta)]
  J=-(1/m)*sum(y*log(predictions)+(1-y)*log(1-predictions))+(lambda/(2*m))*sum(theta_reg^2)
  return(J)
  
}