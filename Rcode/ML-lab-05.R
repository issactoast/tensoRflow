# ML lab 05: Logistic regression
# Source: https://youtu.be/2FeWGgnyLSw?list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm

library(tensorflow)

# Given data
X.data <- cbind(1:6, c(2,3,1,3,3,2))
X.data <- cbind(1, X.data)
X.data

y.data <- matrix(c(0, 0, 0, 1, 1, 1), ncol = 1)
y.data

# Step1: Build graph
# Set the variables in tensorflow
X <- tf$placeholder(tf$float32, shape(6L, 3L))
y <- tf$placeholder(tf$float32, shape(6L, 1L))

# W vector is our parameter to be estimated.
# Here W include the parameter b in the lecture.
W <- tf$Variable(tf$constant(c(1.0, 1.0, 1.0),
                                shape = c(3L, 1L)),
                    name = 'parameter')

# Calculate y_hat
# sigmoid(x) = 1/(1 + exp(-x))
y_hat <- tf$sigmoid(tf$matmul(X, W))

# Cost function
cost <- -tf$reduce_mean(y * tf$log(y_hat) +
                          (1 - y) * tf$log(1 - y_hat))

# Gradient Descent
optimizer <- tf$train$GradientDescentOptimizer(learning_rate = 0.1)
train <- optimizer$minimize(cost)

sess <- tf$Session()
sess$run(tf$global_variables_initializer())

# Training
result <- matrix(0, nrow = 5000, ncol = 4)
for (i in 1:5000) {
  result[i,] <- unlist(sess$run(c(cost, W, train),
                                feed_dict = dict(X = X.data, y = y.data)))
}
result <- cbind(1:5000, result)
colnames(result) <- c("step", "cost", "b",
                      "W1", "W2")
# Check the result
head(result)
tail(result)

# Check the cost decreasing
plot(1:5000, result[1:5000,2])

# Check our result
W.est <- as.vector(result[5000,3:5])

# Calculate result
round(sess$run(tf$sigmoid(X.data %*% W.est)))
y.data

# Close session
sess$close()
