#-------------------------------------------------------------------------------
# Sistemas Inteligentes para Gestión de la Empresa
# Curso 2016-2017
# Departamento de Ciencias de la Computación e Inteligencia Artificial
# Universidad de Granada
#
# Juan Gómez-Romero (jgomez@decsai.ugr.es)
# Francisco Herrera Trigueros (herrera@decsai.ugr.es)
#
# Example of MNIST with mxnetR using a simple network topology
# MNIST data is loaded by using the 'darch' library in CRAN
# Implementation based on: 
# https://www.r-bloggers.com/image-recognition-tutorial-in-r-using-deep-convolutional-neural-networks-mxnet-package/ 
#-------------------------------------------------------------------------------

# Clear workspace
rm(list=ls())

#-------------------------------------------------------------------------------
# Load and pre-process images
#-------------------------------------------------------------------------------

# Load libraries
library(darch)

# Load images 
doNotCreateRData <- FALSE           # change to TRUE after first run

if(!doNotCreateRData) {
  readMNIST("./mnist-digits/") 
}
load("./mnist-digits/train.RData") # trainData + trainLabels
load("./mnist-digits/test.RData")  # testData + testLabels

# Display first image
# img <- Image(matrix(trainData[1, ], nrow = 28, ncol = 28))
# display(img, method="raster")

#-------------------------------------------------------------------------------
# Setup mxnetR
#-------------------------------------------------------------------------------

# Install mxnetR, see: https://github.com/dmlc/mxnet/tree/master/R-package
# (uncomment on first run)
# install.packages("drat", repos="https://cran.rstudio.com")
# drat:::addRepo("dmlc")
# install.packages("mxnet")

# Load MXNet
require(mxnet)
  
#-------------------------------------------------------------------------------
# Prepare training and validation sets
#-------------------------------------------------------------------------------
# Watch out the t(), mxnetR uses data in column format

# Build training matrix
train_x <- t(trainData) 
train_y <- apply(trainLabels, MARGIN=1, which.max) - 1
train_array <- train_x 
dim(train_array) <- c(28, 28, 1, ncol(train_x))

# Build validation matrix
test_x <- t(testData)
test_y <- apply(testLabels, MARGIN=1, which.max) - 1
test_array <- test_x
dim(test_array) <- c(28, 28, 1, ncol(test_x))

#-------------------------------------------------------------------------------
# Set up the symbolic model
#-------------------------------------------------------------------------------
# This is a very simple network topology (2 CONV+RELU+POOL layers + 2 FC layers)

data <- mx.symbol.Variable('data')
# 1st convolutional layer
conv_1 <- mx.symbol.Convolution(data = data, kernel = c(5, 5), num_filter = 20)
tanh_1 <- mx.symbol.Activation(data = conv_1, act_type = "tanh")
pool_1 <- mx.symbol.Pooling(data = tanh_1, pool_type = "max", kernel = c(2, 2), stride = c(2, 2))
# 2nd convolutional layer
conv_2 <- mx.symbol.Convolution(data = pool_1, kernel = c(5, 5), num_filter = 50)
tanh_2 <- mx.symbol.Activation(data = conv_2, act_type = "tanh")
pool_2 <- mx.symbol.Pooling(data=tanh_2, pool_type = "max", kernel = c(2, 2), stride = c(2, 2))
# 1st fully connected layer
flatten <- mx.symbol.Flatten(data = pool_2)
fc_1 <- mx.symbol.FullyConnected(data = flatten, num_hidden = 500)
tanh_3 <- mx.symbol.Activation(data = fc_1, act_type = "tanh")
# 2nd fully connected layer
fc_2 <- mx.symbol.FullyConnected(data = tanh_3, num_hidden = 10)
# Output. Softmax output since we'd like to get some probabilities.
NN_model <- mx.symbol.SoftmaxOutput(data = fc_2)

#-------------------------------------------------------------------------------
# Pre-training set up
#-------------------------------------------------------------------------------

# Set seed for reproducibility
mx.set.seed(100)

# Device used. CPU in my case (using the R version)
# For GPU, use mx.gpu() - see: https://github.com/dmlc/mxnet/issues/5052 
devices <- mx.cpu()

#-------------------------------------------------------------------------------
# Training
#-------------------------------------------------------------------------------

# Train the model
model <- mx.model.FeedForward.create(NN_model,
                                     X = train_array,
                                     y = train_y,
                                     ctx = devices,
                                     num.round = 5,
                                     array.batch.size = 100,
                                     learning.rate = 0.01,
                                     momentum = 0.9,
                                     eval.metric = mx.metric.accuracy,
                                     epoch.end.callback = mx.callback.log.train.metric(100))

saveRDS(object = model, file = "MNISTmodel_01.rds") # save model for further use

#-------------------------------------------------------------------------------
# Validation with test dataset
#-------------------------------------------------------------------------------

# Predict labels
predicted <- predict(model, test_array)
# Assign labels
predicted_labels <- max.col(t(predicted)) - 1
# Get accuracy
class_table <- table(test_y, predicted_labels)
sum(diag(class_table))/10000
