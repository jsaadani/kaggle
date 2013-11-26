sigmoidGradient <- function(z){
  
  source("sigmoid.R")
  g <- sigmoid(z)*(1-sigmoid(z))
  return(g)
}