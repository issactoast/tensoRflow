# ML-lab-10-2: MNIST Data revisit
# Source: https://youtu.be/6CCXyfvubvY?list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm

library(tensorflow)
library(tidyverse)

# The MNIST Data
datasets <- tf$contrib$learn$datasets
mnist <- datasets$mnist$read_data_sets("MNIST-data", one_hot = TRUE)

### Define Constants
tf$reset_default_graph()
X <- tf$placeholder(tf$float32, shape(NULL, 784L))
y <- tf$placeholder(tf$float32, shape(NULL, 10L))
kprob <- tf$placeholder(tf$float32)

### Deeper layers with dropout
## Layer 1
W1 <- tf$get_variable("W1", shape = shape(784L, 512L),
                      initializer = tf$contrib$layers$xavier_initializer())
b1 <- tf$Variable(tf$random_normal(shape = shape(512L)))
L1 <- tf$nn$relu(tf$matmul(X, W1) + b1) %>% tf$nn$dropout(keep_prob = kprob)

## Layer2
W2 <- tf$get_variable("W2", shape = shape(512L, 512L),
                      initializer = tf$contrib$layers$xavier_initializer())
b2 <- tf$Variable(tf$random_normal(shape = shape(512L)))
L2 <- tf$nn$relu(tf$matmul(L1, W2) + b2) %>% tf$nn$dropout(keep_prob = kprob)

## Layer3
W3 <- tf$get_variable("W3", shape = shape(512L, 512L),
                      initializer = tf$contrib$layers$xavier_initializer())
b3 <- tf$Variable(tf$random_normal(shape = shape(512L)))
L3 <- tf$nn$relu(tf$matmul(L2, W3) + b3) %>% tf$nn$dropout(keep_prob = kprob)

## Layer4
W4 <- tf$get_variable("W4", shape = shape(512L, 512L),
                      initializer = tf$contrib$layers$xavier_initializer())
b4 <- tf$Variable(tf$random_normal(shape = shape(512L)))
L4 <- tf$nn$relu(tf$matmul(L3, W4) + b4) %>% tf$nn$dropout(keep_prob = kprob)

## Layer5
W5 <- tf$get_variable("W5", shape = shape(512L, 10L),
                      initializer = tf$contrib$layers$xavier_initializer())
b5 <- tf$Variable(tf$random_normal(shape = shape(10L)))


### Hypothesis
y_hat <- tf$nn$softmax(tf$matmul(L4, W5) + b5)
cost <- tf$nn$softmax_cross_entropy_with_logits(
  logits = tf$matmul(L4, W5) + b5,
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
                  feed_dict = dict(X = batch_xs, y = batch_ys, kprob = 0.7))
    avg_cost <- avg_cost + sum(c[[1]] / total_batch)
  }
  cat("epoch ", epoch,": cost = ", avg_cost, "\n")
}

## check the ACC
correct_prediction <- tf$equal(tf$argmax(y_hat, 1L), tf$argmax(y, 1L))
accuracy <- tf$reduce_mean(tf$cast(correct_prediction, tf$float32))
sess$run(accuracy, feed_dict = dict(X = mnist$test$images,
                                    y = mnist$test$labels,
                                    kprob = 1 ))

## Tuned parameter
# W <- sess$run(c(W1, W2, W3))
# b <- sess$run(c(b1, b2, b3))

# Close session
sess$close()

