# ML lab 04-1: Multi-variable linear regression
# Source: https://youtu.be/Y0EF9VqRuEA?list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm

library(tensorflow)
tf$reset_default_graph()

# Given data
X.data <- matrix(c(73, 80, 75,
                   93, 88, 93,
                   89, 91, 90,
                   96, 98, 100,
                   73, 66, 70), ncol = 3, byrow = TRUE)
X.data <- cbind(1, X.data)
y.data <- matrix(c(152, 185, 180, 196, 142), ncol = 1)

# Step1: Build graph
# Set the variables in tensorflow
X <- tf$placeholder(tf$float32, shape(5L, 4L))
y <- tf$placeholder(tf$float32, shape(5L, 1L))

# W is a slope and  b is an intercept.
beta <- tf$Variable(tf$constant(c(1.0, 1.0, 1.0, 1.0),
                                shape = c(4L, 1L)),
                                name = 'parameter')

# Calculate y_hat
y_hat <- tf$matmul(X, beta)

# Cost function
cost <- tf$reduce_sum(tf$square(y - y_hat))

# Set the Cost function
gradient <- tf$reduce_mean(-tf$matmul(X,(y - y_hat), transpose_a = TRUE ))
decent <- beta - 1e-5 * gradient
update <- beta$assign(decent)


sess <- tf$Session()
sess$run(tf$global_variables_initializer())

# Training
result <- matrix(0, nrow = 20, ncol = 5)
for (i in 1:20) {
  result[i,] <- unlist(sess$run(c(cost, update),
                                feed_dict = dict(X = X.data, y = y.data)))
}
result <- cbind(1:20, result)
colnames(result) <- c("step", "cost", "intercept",
                      "beta1", "beta2", "beta3")
head(result)
plot(result[,"step"], result[,"cost"])

# Check our result
beta.est <- as.vector(result[20,3:6])

#Calculate y hat & compare y.data
X.data %*% beta.est
y.data

# Close session
sess$close()

