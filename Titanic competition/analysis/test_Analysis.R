data <- read.csv('train.csv')

head(data)
str(data)
attach(data)

#visualization
hist(age)
tail(data,10)

data_train <- data$survived

