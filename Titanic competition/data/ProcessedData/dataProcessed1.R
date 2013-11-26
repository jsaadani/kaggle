data <- read.csv("./data/train.csv")

#perform subsampling
trainIndicator=rbinom(nrow(data),size=1,prob=0.7)
trainData <- data[trainIndicator==1,]#70% of the data
validatonData <- data[trainIndicator==0,]#30% of the data

#replace missing embarked values (2) with C (most common)
trainData[!(trainData$embarked %in% ports),c("embarked")] <- "C"







#exploratory analysis
plot(trainData[,c("age","fare")])
head(trainData)
summary(data$pclass)
quantile(trainData$age,na.rm=TRUE)
hist(trainData$age,breaks=16)
child <- trainData[trainData$age<18,]
sum(is.na(trainData))
sum(is.na(validatonData))
plot(pclass,survived,pch=19,col=sex,
     ylim=c(0,5))
plot()

str(trainData)
table(trainData$sibsp)
quantile(trainData$sibsp)
attach(trainData)
cor(age,survived)
any(age<0, na.rm=TRUE)
range(age, na.rm=TRUE)
range(fare)
colMeans(trainData)

embarked <- gsub(" ","S",embarked)

levels(embarked)[1]
trainData$embarked[which(embarked=="")] <- "C"
levels(embarked)
table(trainData$embarked)

trainData$embarked[580] <- "C"
levels(trainData$embarked)
table(trainData$embarked)
which(trainData$emabrked==())

ports <- c("C","Q","S")
trainData[!(trainData$embarked %in% ports),c("embarked")] <- "C"
table(trainData$embarked)