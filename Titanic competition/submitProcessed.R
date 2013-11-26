data <- read.csv("./data/test.csv")

#source("predictions2.R")
load("predictions_NN_1.rda")
submit <- cbind(predictions,data)[,1:2]
names(submit) <- c("Survived","PassengerId")
submit <- submit[,c("PassengerId","Survived")]
write.csv(submit,file="submit6.csv",row.names=FALSE)