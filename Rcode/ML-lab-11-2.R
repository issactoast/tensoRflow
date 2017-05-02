# ML-lab-11-2: MNIST Data with CNN
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
kprob <- tf$placeholder(tf$float32)

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
L1 <- max_pool_2x2(tf$nn$relu(L1)) %>% 
      tf$nn$dropout(keep_prob = kprob)


# Layer 2
W2 <- W.gen(shape(3L, 3L, 32L, 64L))
L2 <- conv2d(L1, W2)
L2 <- max_pool_2x2(tf$nn$relu(L2)) %>% 
      tf$nn$dropout(keep_prob = kprob)

# Layer 3
W3 <- W.gen(shape(3L, 3L, 64L, 128L))
L3 <- conv2d(L2, W3)
L3 <- max_pool_2x2(tf$nn$relu(L3)) %>% 
      tf$nn$dropout(keep_prob = kprob)
L3_flat <- tf$reshape(L3, shape(-1L, 4L * 4L * 128L))

# Layer 4 Fully Connected 1
W4 <- tf$get_variable("W4", shape = shape(4L * 4L * 128L, 625L),
                      initializer = tf$contrib$layers$xavier_initializer())
b1  <- b.gen(shape(625L))
L4 <- tf$nn$relu(tf$matmul(L3_flat, W4) + b1) %>% 
      tf$nn$dropout(keep_prob = kprob)

# Layer 5 FC2
W5 <- tf$get_variable("W5", shape = shape(625L, 10L),
                      initializer = tf$contrib$layers$xavier_initializer())
b2  <- b.gen(shape(10L))
y_hat <- tf$nn$softmax(tf$matmul(L4, W5) + b2)

# Cost function
cost <- tf$nn$softmax_cross_entropy_with_logits(
  logits = tf$matmul(L4, W5) + b2,
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
#sess$run(accuracy, feed_dict = dict(X = mnist$test$images,
#                                    y = mnist$test$labels,
#                                    kprob = 1 ))

# Running out of Memory..
acc <- 0
for (i in 0:9){
  acc <- acc + 
    1000 * sess$run(accuracy, feed_dict = dict(
      X = mnist$test$images[(1 + (i * 1000)):(1000 + (i * 1000)),],
      y = mnist$test$labels[(1 + (i * 1000)):(1000 + (i * 1000)),],
      kprob = 1 ))
}

# Accurary
acc / 10000


# Falsed prediction
smalltest_img <- mnist$test$images[1:1000, ]
smalltest_lab <- mnist$test$labels[1:1000, ]
correct_prediction <- tf$equal(tf$argmax(y_hat, 1L), tf$argmax(y, 1L))
corr_index <- sess$run(correct_prediction, feed_dict = dict(
                  X = smalltest_img,
                  y = smalltest_lab,
                  kprob = 1 ))
false_Xdata <- smalltest_img[!corr_index,]
false_ydata <- smalltest_lab[!corr_index,]

## Let us take a look at the data which is NOT correctly prediction.
mat.plot <- function(mat, ...){
  image(t(apply(mat, 2, rev)),
        col  = gray((0:255)/255),
        axes = F, ...)
}

check <- function(num){
  sample <- matrix(1 - false_Xdata[num,], nrow = 28, byrow = TRUE)
  pred <- sess$run(tf$argmax(y_hat, 1L),
                   feed_dict=dict( 
                     X = matrix(false_Xdata[num,], nrow = 1),
                     kprob = 1 ))
  ans <- which.max(false_ydata[num,]) - 1L
  mat.plot(sample, main = paste( ans, "predicted", pred))
}

# number of falsed pred. out of 1000
dim(false_Xdata)[1]

# check the result
par(mfrow=c(2, 2), mar=c(0,0,1,0))
for (i in 1:4) check(i)

# Close session
sess$close()
