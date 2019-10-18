# ML lab 06-01,02: Softmax Regression
# Source: https://youtu.be/VRnubDzIy3A?list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm
# Source: https://youtu.be/E-io76NlsqA?list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm

library(tensorflow)

# Given data
X.data <- rbind(c(1, 2, 1, 1),
                c(2, 1, 3, 2),
                c(3, 1, 3, 4),
                c(4, 1, 5, 5),
                c(1, 7, 5, 5),
                c(1, 2, 5, 6),
                c(1, 6, 6, 6),
                c(1, 7, 7, 7))
X.data <- cbind(1, X.data)

y.data <- matrix(0, ncol = 3, nrow = 8)
y.data[1:3,1] <- 1
y.data[4:6,2] <- 1
y.data[7:8,3] <- 1

y.data

# plot(X.data[,2], X.data[,3],
#      pch = y.data,
#      col = c("red", "blue", "green")[rep(c(1,2,3), each = 3)])

# We think our model is muti-variate regression, i.e.,
# y ~ X %*% beta + e
# y is 8 by 3 matrix
# X is 8 by 5 matrix (including intercept)
# beta is 5 by 3 matrix 

# Step1: Build graph
# Set the variables in tensorflow
X <- tf$placeholder(tf$float32, shape(8L, 5L))
y <- tf$placeholder(tf$float32, shape(8L, 3L))


# beta vector is our parameter to be estimated.
# Here beta is equal to the W vector and also include
# the parameter b in the lecture.
# beta <- tf$Variable(tf$zeros(shape(5L, 3L)),
#                     name  = 'parameter')
beta <- tf$Variable(tf$constant(rep(0.0, 15),
                    shape = c(5L, 3L)),
                    name = 'parameter')

# Calculate y_hat
# Note: for x as a n by 1 vector,
# the element of softmax function of x is as follows:
# softmax(x)_i = exp(x_i)/sum(exp(x_i)) for i = 1,...,n
y_hat <- tf$nn$softmax(tf$matmul(X, beta))

# Cost function: cross_entropy
cross_entropy <- tf$reduce_mean(-tf$reduce_sum( y * tf$log(y_hat), axis = 1L))
cross_entropy <- tf$nn$softmax_cross_entropy_with_logits(logits = tf$matmul(X, beta),
                                                         labels = y)

optimizer <- tf$train$GradientDescentOptimizer(learning_rate = 0.2)
train <- optimizer$minimize(cross_entropy)

sess <- tf$Session()
sess$run(tf$global_variables_initializer())

# Training
for (step in 1:10000) {
  sess$run(train, feed_dict = dict(X = X.data, y = y.data))
  if (step %% 200 == 0)
  cat("step", step, "-", sess$run(cross_entropy,
                                  feed_dict = dict(X = X.data, y = y.data)), "\n")
}

beta_hat <- sess$run(beta)

# Fitted prediction values:
# Note the python has zero base system
sess$run( tf$argmax(y_hat, axis = 1L), feed_dict = dict(X = X.data)) + 1


# Close session
sess$close()
