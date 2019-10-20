# ML lab 07-02: MNIST Data 
# Source: https://youtu.be/ktd5yrki_KA?list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm

# data preparation
# temp_m <- matrix(0, ncol = 10, nrow = 60000)
# for(i in 1:60000){
#     temp_m[i, mnist_data.trainy[i] + 1] <- 1 
# }
# mnist_data.trainy <- temp_m
# head(mnist_data.trainy)
# colnames(mnist_data.trainy) <- paste0("num", 0:9)
# write.csv(mnist_data.trainy, "./MNIST-data/trainy.csv", row.names = FALSE)
# dim(mnist_data.trainX)
# head(mnist_data.trainX[,1:20])
# write.csv(mnist_data.trainX, "./MNIST-data/trainx.csv", row.names = FALSE)
# mnist_data.trainX <- mnist_data.trainX[,-1]


library(tensorflow)

tf$reset_default_graph()

# The MNIST Data
mnist_data.trainX <- read.csv("./MNIST-data/trainx.csv")
mnist_data.trainy <- read.csv("./MNIST-data/trainy.csv")
mnist_data.testX <- read.csv("./MNIST-data/testx.csv")
mnist_data.testy <- read.csv("./MNIST-data/testy.csv")

mnist_data.trainX <- as.matrix(mnist_data.trainX)
mnist_data.trainy <- as.matrix(mnist_data.trainy)

temp_m <- matrix(0, ncol = 10, nrow = 10000)
for(i in 1:10000){
    temp_m[i, mnist_data.testy[i,] + 1] <- 1
}
mnist_data.testy <- temp_m

# mnist_data.testX <- as.matrix(mnist_data.testX)/255
# mnist_data.testy <- as.matrix(mnist_data.testy)/255


# data visualization
viz_mnist <- function(sample_vec){
    ### MNIST data image of 784 vector -> 1 by 784
    sample <- matrix(as.numeric(sample_vec), nrow = 28, byrow = TRUE)
    grays  <- rgb(red = 0:255/255, blue = 0:255/255, green = 0:255/255)
    rotate <- function(x) apply(t(x), 2, rev)
    heatmap(rotate(sample), Rowv=NA, Colv=NA,
            labRow = FALSE, labCol = FALSE,
            col = grays, scale = "none")    
}

viz_mnist(mnist_data.trainX[2,])

# Building graph
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

train_epochs <- 300L
batch_size <- 1000L

for (epoch in 1:train_epochs) {
  avg_cost <- 0
  total_batch <- as.integer(dim(mnist_data.trainX)[1] / batch_size)
  
  for (j in 1:total_batch){
    batches  <- batch_size * (j - 1) + (1L:1000L)
    batch_xs <- mnist_data.trainX[batches,]
    batch_ys <- mnist_data.trainy[batches,]
    c <- sess$run(c(cross_entropy, train),
                  feed_dict = dict(X = batch_xs, y = batch_ys))
    avg_cost <- avg_cost + sum(c[[1]] / total_batch)
  }
  cat("epoch ", epoch,": cost = ", avg_cost, "\n")
}

## check the ACC
correct_prediction <- tf$equal(tf$argmax(y_hat, 1L), tf$argmax(y, 1L))
accuracy <- tf$reduce_mean(tf$cast(correct_prediction, tf$float32))
sess$run(accuracy, feed_dict=dict(X = mnist_data.testX, y = mnist_data.testy))

## Tuned parameter
W <- sess$run(W)
b <- sess$run(b)
dim(W)
dim(b)
dim(sample_m)
## Let us take a look at the data.
check <- function(num){
  sample_m <- matrix(as.numeric(mnist_data.testX[3,]), nrow = 28, byrow = TRUE)
  grays = rgb(red = 0:255/255, blue = 0:255/255, green = 0:255/255)
  rotate <- function(x) apply(t(x), 2, rev)
  pred <- which.max(as.numeric(t(as.vector(sample_m) %*% W)) + b) - 1
  heatmap(rotate(sample_m), Rowv=NA, Colv=NA,
          labRow = FALSE, labCol = FALSE,
          col = grays, scale = "none",
          main = paste("predicted as", pred))
}
check(6)


# Close session
sess$close()
