# Lab 2
# Source: <iframe width="560" height="315" src="https://www.youtube.com/embed/mQGwjrStQgg?list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm" frameborder="0" allowfullscreen></iframe>

library(tensorflow)

# Step1: Build graph

# X and y data
# Let's assume our data determined by the
# parameter 2 and 3, but noise hides that.
x <- as.numeric(c(1:10))
y <- 2 * x + 3 + rnorm(10, mean = 0, sd = 0.5)

# Let's see what our data looks like?
plot(x, y, xlim = c(0, 10), ylim = c(0, 30))

# Set the variables in tensorflow
# W is a slope and  b is an intercept.
W <- tf$Variable(tf$constant(1.0), name = 'weight')
b <- tf$Variable(tf$constant(2.0), name = 'weight')

# Calculate y_hat
y_hat <- x * W + b

# Set the Cost function
cost <- tf$reduce_mean(tf$square(y - y_hat))

# Gradient Descent
optimizer <- tf$train$GradientDescentOptimizer(learning_rate = 0.01)
train <- optimizer$minimize(cost)

sess <- tf$Session()
sess$run(tf$global_variables_initializer())

# Training
for (i in 1:3000) {
  sess$run(train)
  if (i %% 200 == 0 ) print( c(sess$run(W), sess$run(b)) )
}

# Check our answer
sess$run(W)
sess$run(b)
abline(sess$run(b), sess$run(W))

# Close session
sess$close()
