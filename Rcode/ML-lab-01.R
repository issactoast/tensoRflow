# Lecture Youtube
# Source: https://youtu.be/-57Ne86Ia8w?list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm

library(tensorflow)

# Check the version of Tensorflow
tf$VERSION

# Say hello to your tensorflow
hello <- tf$constant("Hello, Tensorflow")

sess <- tf$Session()
print(sess$run(hello))
sess$close()


# Add number node
node1 <- tf$constant(3.0, tf$float32)
node2 <- tf$constant(5.0)
node3 <- tf$add(node1, node2)

print(node2)
sess <- tf$Session()
sess$run(node2)
sess$close()


# Placeholder
a <- tf$placeholder(tf$float32)
b <- tf$placeholder(tf$float32)
adder.node <- a + b


# Using this, sess will automatically closed.
with(tf$Session() %as% sess, {
  print(sess$run(adder.node,
                 feed_dict = dict(a = 3, b = 4.5)))
  print(sess$run(adder.node,
                 feed_dict = dict(a = c(1:10), b = c(21:30))))
})
