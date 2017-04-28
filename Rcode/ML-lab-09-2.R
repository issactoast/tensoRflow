# tensoRflow ML-lab-09-2: Tensor board
# Sung Kim 교수님 강의 소스: https://youtu.be/oFGHOsAYiz0?list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm

library(tensorflow)
library(reticulate)

# Data preparation
X.data <- r_to_py(matrix(c(0,0,0,1,1,0,1,1), ncol = 2, byrow = TRUE))
y.data <- r_to_py(matrix(c(0,1,1,0), ncol = 1))

# Declare placeholder
X <- tf$placeholder(tf$float32, shape(4L, 2L))
y <- tf$placeholder(tf$float32, shape(4L, 1L))

# Combine layer1
with(tf$name_scope("layer1") %as% scope, {
  # Define variables
  W1 <- tf$Variable(tf$random_normal(c(2L, 2L)), name = "weight1")
  b1 <- tf$Variable(tf$random_normal(c(1L, 2L)), name = "bias1")
  layer1 <- tf$sigmoid(tf$matmul(X, W1) + b1)

  W1_hist <- tf$summary$histogram("weight1", W1)
  b1_hist <- tf$summary$histogram("bias1", b1)
  layer1_hist <- tf$summary$histogram("layer1", layer1)
})

with(tf$name_scope("layer2") %as% scope, {
  # Define variables
  W2 <- tf$Variable(tf$random_normal(c(2L, 1L)), name = "weight2")
  b2 <- tf$Variable(tf$random_normal(c(1L, 1L)), name = "bias2")
  y.hat <- tf$sigmoid(tf$matmul(layer1, W2) + b2)
  
  W2_hist <- tf$summary$histogram("weight1", W2)
  b2_hist <- tf$summary$histogram("bias1", b2)
  layer2_hist <- tf$summary$histogram("layer2", y.hat)
})

with(tf$name_scope("cost") %as% scope, {
  # cost/loss function
  cost <- -tf$reduce_mean(y * tf$log(y.hat) +
                            (1 - y) * tf$log(1 - y.hat))
  cost_sum <- tf$summary$scalar("cost", cost)  
})

with(tf$name_scope("train") %as% scope, {
  train <- tf$train$GradientDescentOptimizer(learning_rate = 0.1)$minimize(cost)
})

# Prediction & Accuracy
prediction <- tf$cast(y.hat > 0.5, dtype = tf$float32)
acc <- tf$reduce_mean(tf$cast(tf$equal(prediction, y), dtype = tf$float32))
acc_sum <- tf$summary$scalar("accuracy", acc)



# Start Session
sess <- tf$Session()
saver = tf$train$Saver()
sess$run(tf$global_variables_initializer())

merged_summary <- tf$summary$merge_all()
writer <- tf$summary$FileWriter("./logs/xor_logs_r0_01")
writer$add_graph(sess$graph)

# Training
for (step in 1:10000) {
  #sess$run(train, feed_dict = dict(X = X.data, y = y.data))
  summary <- sess$run(c(merged_summary, train), feed_dict = dict(X = X.data, y = y.data))
  writer$add_summary(summary[[1]], global_step = step)
  if (step %% 100 == 0)
    cat("step", step, "-",
        sess$run(cost, feed_dict = dict(X = X.data, y = y.data)),
        sess$run(W1), sess$run(W2),
        "\n")
}

# Accurary report
result <- sess$run(c(y.hat, prediction, acc),
                   feed_dict = dict(X = X.data, y = y.data) )
cat(" y.hat      = ", result[[1]], "\n",
    "Prediction = ", result[[2]], "\n",
    "Accuracy   = ", result[[3]], "\n")

# close session
sess$close()
