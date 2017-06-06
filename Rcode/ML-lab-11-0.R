# ML-lab-11-1: MNIST Data with CNN
# Lecture Source: https://youtu.be/E9Xh_fc9KnQ?list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm

library(tensorflow)
library(tidyverse)

mat.plot <- function(mat, ...){
  image(t(apply(mat, 2, rev)),
        col  = gray((0:255)/255),
        axes = F, ...)
}
toy.image <- matrix(as.numeric(c(1:9)), ncol = 3, byrow = TRUE)
mat.plot(toy.image)

x <- tf$placeholder(tf$float64, shape(3L, 3L))
x_image <- tf$reshape(x, shape(1L, 3L, 3L, 1L))
weight <- tf$constant(matrix(as.numeric(rep(1,4)), ncol = 2),
                      shape = shape(2L, 2L, 1L, 1L),
                      dtype = "float64")
conv2d <- tf$nn$conv2d(x_image, weight,
                       strides = c(1L,1L,1L,1L),
                       padding = 'VALID')
conv2d_img <- conv2d$eval(feed_dict = dict(x = toy.image))
sess$close()
