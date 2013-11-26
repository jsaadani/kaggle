data <- read.csv("train.csv")
str(data)
head(data)

range(data$age, na.rm=TRUE)
range(data$fare)

sum(is.na(data$age))
summary(data)
sapply(data[1,],class)
unique(data$embarked)
table(data$sex,data$survived, useNA="ifany")
sd(data$age, na.rm=TRUE)

#female
233/(233+81)
#male
109/(468+109)


missingAge <- data[is.na(data$age),]
head(missingAge)
summary(missingAge)

177/891
#====missing data====
goodData <- complete.cases(data)
cData <- data[goodData,]
summary(cData$embarked)

dataTrain <- cData[,c("survived","pclass","sex","age","sibsp","parch","fare","embarked")]


tail(dataTrain$embarked,30)
summary(dataTrain)

str(dataTrain)

toNormalize
sapply(dataTrain,sd)

matrice <- as.matrix(dataTrain)

matrice[1:5,]

table(data$age,embarked,useNA="ifany")
hist(data$agedata$sex)
as.numeric(dataTrain$sex)
dataTrain$sex

dens <- density(data$age, na.rm=TRUE)
densMales <- density(data$age[which(data$sex=="male")],na.rm=TRUE)
densFemales <- density(data$age[which(data$sex=="female")],na.rm=TRUE)
plot(dens,lwd=3,col="blue") 
lines(densMales,lwd=3,col="orange")
lines(densFemales,lwd=3,col="pink")

source("featureNormalize.R")

featureNormalize(dataTrain[,c("age")])

X <- dataTrain[,c("age","fare")]
featureNormalize(X)

dataTrain[,c("age","fare")] <- featureNormalize(dataTrain[,c("age","fare")])

head(dataTrain)

head(as.numeric(dataTrain$sex)-1)
range(as.numeric(dataTrain$embarked))

dataTrain$sex <- as.numeric(dataTrain$sex)-1
dataTrain$embarked <- as.numeric(dataTrain$embarked)

str(dataTrain)
dump("dataTrain",file="data.R")
data <- read.csv("data.R")
data <- load("data.R")
source("data.R")

