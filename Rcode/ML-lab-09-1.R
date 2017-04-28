# tensoRflow ML-lab-09-1: Double layers
# Sung Kim 교수님 강의 소스: https://youtu.be/oFGHOsAYiz0?list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm

library(tensorflow)
library(reticulate)

# Data preparation
X.data <- r_to_py(matrix(c(0,0,0,1,1,0,1,1), ncol = 2, byrow = TRUE))
y.data <- r_to_py(matrix(c(0,1,1,0), ncol = 1))

# Declare placeholder
X <- tf$placeholder(tf$float32, shape(4L, 2L))
y <- tf$placeholder(tf$float32, shape(4L, 1L))

# Define variables
W1 <- tf$Variable(tf$random_normal(c(2L, 2L)), name = "weight1")
b1 <- tf$Variable(tf$random_normal(c(1L, 2L)), name = "bias1")
layer1 <- tf$sigmoid(tf$matmul(X, W1) + b1)

W2 <- tf$Variable(tf$random_normal(c(2L, 1L)), name = "weight2")
b2 <- tf$Variable(tf$random_normal(c(1L, 1L)), name = "bias2")
y.hat <- tf$sigmoid(tf$matmul(layer1, W2) + b2)

# cost/loss function
cost <- -tf$reduce_mean(y * tf$log(y.hat) + (1 - y) * tf$log(1 - y.hat))
train <- tf$train$GradientDescentOptimizer(learning_rate = 0.1)$minimize(cost)

# Launch graph
sess <- tf$Session()
sess$run(tf$global_variables_initializer())

# Training
for (step in 1:10000) {
  sess$run(train, feed_dict = dict(X = X.data, y = y.data))
  if (step %% 500 == 0)
    cat("step", step, "-",
        sess$run(cost, feed_dict = dict(X = X.data, y = y.data)),
        "\n")
}

# Prediction & estimated parameter
sess$run(tf$cast(y.hat > 0.5, dtype = tf$float32), feed_dict = dict(X = X.data, y = y.data))
sess$run(c(W1, W2))
sess$run(c(b1, b2))

# close session
sess$close()