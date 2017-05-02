# ML-lab-11-1: MNIST Data with CNN
# Lecture Source: https://youtu.be/E9Xh_fc9KnQ?list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm

library(tensorflow)
library(tidyverse)

# The MNIST Data
datasets <- tf$contrib$learn$datasets
mnist <- datasets$mnist$read_data_sets("MNIST-data", one_hot = TRUE)

### Define Constants
tf$reset_default_graph()
X <- tf$placeholder(tf$float32, shape(NULL, 784L))
X_image <- tf$reshape(X, shape(-1L, 28L, 28L, 1L))
y <- tf$placeholder(tf$float32, shape(NULL, 10L))


# Define functions for CNN
W.gen <- function(shape) {
  initial <- tf$truncated_normal(shape, stddev=0.1)
  tf$Variable(initial)
}

b.gen <- function(shape) {
  initial <- tf$constant(0.1, shape = shape)
  tf$Variable(initial)
}

conv2d <- function(x, W) {
  tf$nn$conv2d(x, W, strides=c(1L, 1L, 1L, 1L), padding='SAME')
}

max_pool_2x2 <- function(x) {
  tf$nn$max_pool(x, 
    ksize=c(1L, 2L, 2L, 1L),
    strides=c(1L, 2L, 2L, 1L), 
    padding='SAME')
}


# Layer 1
W1 <- W.gen(shape(3L, 3L, 1L, 32L))
L1 <- conv2d(X_image, W1)
L1 <- max_pool_2x2(tf$nn$relu(L1))


# Layer 2
W2 <- W.gen(shape(3L, 3L, 32L, 64L))
L2 <- conv2d(L1, W2)
L2 <- max_pool_2x2(tf$nn$relu(L2)) %>%
      tf$reshape(shape = shape(-1L, 7L * 7L * 64L))

# Layer 3
W3 <- tf$get_variable("W3", shape = shape(7L * 7L * 64L, 10L),
                      initializer = tf$contrib$layers$xavier_initializer())
b <- b.gen(shape(10L))


### Inference
y_hat <- tf$nn$softmax(tf$matmul(L2, W3) + b)
cost <- tf$nn$softmax_cross_entropy_with_logits(
  logits = tf$matmul(L2, W3) + b,
  labels = y
) %>% tf$reduce_mean()

optimizer <- tf$train$AdamOptimizer(learning_rate = 0.001)
train <- optimizer$minimize(cost)

init <- tf$global_variables_initializer()
sess <- tf$Session()
sess$run(init)


train_epochs <- 15L
batch_size <- 100L

set.seed(1111)
for (epoch in 1:train_epochs) {
  avg_cost <- 0
  total_batch <- as.integer(mnist$train$num_examples/ batch_size)
  
  for (j in 1:total_batch){
    batches  <- mnist$train$next_batch(100L)
    batch_xs <- batches[[1]]
    batch_ys <- batches[[2]]
    c <- sess$run(c(cost, train),
                  feed_dict = dict(X = batch_xs, y = batch_ys))
    avg_cost <- avg_cost + sum(c[[1]] / total_batch)
  }
  cat("epoch ", epoch,": cost = ", avg_cost, "\n")
}

## check the ACC
correct_prediction <- tf$equal(tf$argmax(y_hat, 1L), tf$argmax(y, 1L))
accuracy <- tf$reduce_mean(tf$cast(correct_prediction, tf$float32))

# Running out of Memory..
acc <- 0
for (i in 0:9){
  acc <- acc + 
    1000 * sess$run(accuracy, feed_dict = dict(
                    X = mnist$test$images[(1 + (i * 1000)):(1000 + (i * 1000)),],
                    y = mnist$test$labels[(1 + (i * 1000)):(1000 + (i * 1000)),] ))
}

# Accurary
acc / 10000

# Close session
sess$close()
