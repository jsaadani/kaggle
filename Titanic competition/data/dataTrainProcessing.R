data <- read.csv("./data/train.csv")

#====replace NA by mean====
data$age [is.na(data$age)] <- mean(data$age, na.rm=TRUE)

#====Normalize====
source("./functions/featureNormalize.R")
data[,c("age","fare")] <- featureNormalize(data[,c("age","fare")])
data$sex <- as.numeric(data$sex)-1
data$embarked <- as.numeric(data$embarked)

#====split train set and CV set====
# set.seed(2)
# rand <- sample(nrow(data))
# randData <- data[rand,]
# trainSet <- randData[1:623,]
# cvSet <- randData[624:nrow(data),]

trainSet <- data

#====loading data into matrix====
# Xval <- as.matrix(cvSet[,c("pclass","sex","age","sibsp","parch","fare","embarked")])
# yval <- as.matrix(cvSet[,c("survived")])
X <- as.matrix(trainSet[,c("pclass","sex","age","sibsp","parch","fare","embarked")])
y <- as.matrix(trainSet[,c("survived")])

# dump(c("Xval","yval"),file="cvSet.R")
dump(c("X","y"),file="trainSet.R")