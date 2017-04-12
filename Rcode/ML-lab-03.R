# Version 2: Using Placeholders
# Source: <iframe width="560" height="315" src="https://www.youtube.com/embed/mQGwjrStQgg?list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm" frameborder="0" allowfullscreen></iframe>

library(tensorflow)

# Step1: Build graph
# Set the variables in tensorflow
x <- tf$placeholder(tf$float32, shape(10L, 1L))
y <- tf$placeholder(tf$float32, shape(10L, 1L))

# W is a slope and  b is an intercept.
# W <- tf$Variable(tf$zeros(shape(1L, 1L)), name = 'weight')
W <- tf$Variable(tf$constant(1.0, shape = c(1L, 1L)), name = 'weight')
b <- tf$Variable(tf$constant(2.0, shape = c(1L, 1L)), name = 'bias')

# Calculate y_hat
y_hat <- tf$matmul(x, W) + b

# Set the Cost function
cost <- tf$reduce_mean(tf$square(y - y_hat))

# Gradient Descent
optimizer <- tf$train$GradientDescentOptimizer(learning_rate = 0.01)
train <- optimizer$minimize(cost)

sess <- tf$Session()
sess$run(tf$global_variables_initializer())

# Training
x.data <- matrix(c(1:10), ncol = 1)
y.data <- 2 * x.data + 3 + rnorm(10, mean = 0, sd = 0.5)
result <- matrix(0, nrow = 2000, ncol = 3)
for (i in 1:2000) {
  result[i, ] <- unlist(sess$run(c(cost, W, b, train),
                                 feed_dict = dict(x = x.data, y = y.data)) )
  if (i %% 200 == 0 ) print(result[i, 2:3])
}

# Check our answer
# Cost function decreasing
plot(c(1000:2000), result[c(1000:2000),1])

# Let's see what our data looks like?
plot(x.data, y.data, xlim = c(0, 10), ylim = c(0, 30))

sess$run(W)
sess$run(b)
abline(sess$run(b), sess$run(W))

# Close session
sess$close()

