# ML lab 07-02: MNIST Data 
# Source: https://youtu.be/ktd5yrki_KA?list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm

library(tensorflow)

# The MNIST Data
datasets <- tf$contrib$learn$datasets
mnist <- datasets$mnist$read_data_sets("MNIST-data", one_hot = TRUE)

## Let us take a look at the data.

### MNIST data image  28 by 28 -> 1 by 784 vector
str(mnist$train$images)
sample <- matrix(1 - mnist$train$images[4,], nrow = 28) * 255
grays = rgb(red = 0:255/255, blue = 0:255/255, green = 0:255/255)
rotate <- function(x) apply(t(x), 2, rev)
heatmap(rotate(sample), Rowv=NA, Colv=NA,
        labRow = FALSE, labCol = FALSE,
        col = grays, scale = "none")

### X is a placeholder for the 1 by 784 images vector
X <- tf$placeholder(tf$float32, shape(NULL, 784L))
y <- tf$placeholder(tf$float32, shape(NULL, 10L))

### W is 784 by 10 matrix and b is 784 by 1 vector!
### No. of columns is corresponce to the No. of classes: 0 ~ 9
W <- tf$Variable(tf$zeros(shape(784L, 10L)))
b <- tf$Variable(tf$zeros(shape(10L)))

### Hypothesis
y_hat <- tf$nn$softmax(tf$matmul(X, W) + b)

cross_entropy <- tf$nn$softmax_cross_entropy_with_logits(logits = tf$matmul(X, W) + b,
                                                         labels = y)
optimizer <- tf$train$GradientDescentOptimizer(0.5)
train <- optimizer$minimize(cross_entropy)

init <- tf$global_variables_initializer()

sess <- tf$Session()
sess$run(init)

train_epochs <- 15L
batch_size <- 100L

for (epoch in 1:train_epochs) {
  avg_cost <- 0
  total_batch <- as.integer(mnist$train$num_examples/ batch_size)
  
  for (j in 1:total_batch){
    batches  <- mnist$train$next_batch(100L)
    batch_xs <- batches[[1]]
    batch_ys <- batches[[2]]
    c <- sess$run(c(cross_entropy, train),
                  feed_dict = dict(X = batch_xs, y = batch_ys))
    avg_cost <- avg_cost + sum(c[[1]] / total_batch)
  }
  cat("epoch ", epoch,": cost = ", avg_cost, "\n")
}

## check the ACC
correct_prediction <- tf$equal(tf$argmax(y_hat, 1L), tf$argmax(y, 1L))
accuracy <- tf$reduce_mean(tf$cast(correct_prediction, tf$float32))
sess$run(accuracy, feed_dict=dict(X = mnist$test$images, y = mnist$test$labels))

## Tuned parameter
W <- sess$run(W)
b <- sess$run(b)

## Let us take a look at the data.
check <- function(num){
  sample <- matrix(1 - mnist$test$images[num,], nrow = 28)
  grays = rgb(red = 0:255/255, blue = 0:255/255, green = 0:255/255)
  rotate <- function(x) apply(t(x), 2, rev)
  pred <- which.max(as.numeric(t(as.vector(1 - sample) %*% W)) + b) - 1
  heatmap(rotate(sample * 255), Rowv=NA, Colv=NA,
          labRow = FALSE, labCol = FALSE,
          col = grays, scale = "none",
          main = paste("predicted as", pred))
}
check( 310 )
