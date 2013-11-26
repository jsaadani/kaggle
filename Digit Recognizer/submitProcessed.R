load("predictions1.rda")
submitBench <- read.csv("rf_benchmark.csv")
submit <- cbind(submitBench[,1],pred)
colnames(submit) <- c("ImageId","Label")
write.csv(submit,file="submit1.csv",row.names=FALSE)

submit1 <- read.csv("submit1.csv")
str(submit1)