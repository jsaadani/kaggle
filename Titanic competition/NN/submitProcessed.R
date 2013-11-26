data <- read.csv("../data/test.csv")

#source("predictions2.R")
#load vector "pred"
load("predictions_NN_4.rda")
submit <- cbind(pred,data)[,1:2]
names(submit) <- c("Survived","PassengerId")
submit <- submit[,c("PassengerId","Survived")]
write.csv(submit,file="submit_NN_4.csv",row.names=FALSE)

test <- read.csv("submit_NN_4.csv")
head(test)

head(pred)