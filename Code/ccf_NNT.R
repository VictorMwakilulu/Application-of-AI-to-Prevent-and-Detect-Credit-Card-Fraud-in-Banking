install.packages("devtools")
devtools::install_github("rstudio/keras")

library(keras)
install_tensorflow()


cran <- getOption("repos")
cran["dmlc"] <- "https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/R/CRAN/"
options(repos = cran)
install.packages("mxnet")

require(mxnet)
library(caret)
require(mlbench)

nba <- read.csv("~/ANLY 699/Applied Project/Applied-Project/Data/ccf.csv")
names(nba)

#bind variables to use
outcomeName <- 'Outcome'

predictorsNames <- c("dollar", "date", "decision","pos",   "sic")
#

set.seed(1234)
splitIndex <- createDataPartition(nba[,outcomeName], p = .70, list = FALSE, times = 1)
train_nba <- nba[ splitIndex,]
test_nba  <- nba[-splitIndex,]

#"dollar", "date", "decision","pos",   "sic",
predictorsNames <- c("dollar", "date", "decision","pos",   "sic" )
train.x = data.matrix(train_nba[,predictorsNames])
train.y = train_nba[,outcomeName]
test.x = data.matrix(test_nba[, predictorsNames])
test.y = test_nba[,outcomeName]


mx.set.seed(0)
model <- mx.mlp(train.x, train.y, hidden_node=10, out_node=2, out_activation="softmax",
                num.round=20, array.batch.size=15, learning.rate=0.07, momentum=0.9, 
                eval.metric=mx.metric.accuracy)

data <- mx.symbol.Variable("data")
fc1 <- mx.symbol.FullyConnected(data, name="fc1", num_hidden=264)
act1 <- mx.symbol.Activation(fc1, name="relu1", act_type="relu")
fc2 <- mx.symbol.FullyConnected(act1, name="fc2", num_hidden=264)
act2 <- mx.symbol.Activation(fc2, name="relu2", act_type="relu")
fc3 <- mx.symbol.FullyConnected(act1, name="fc3", num_hidden=312)
act3 <- mx.symbol.Activation(fc3, name="relu3", act_type="relu")
fc4 <- mx.symbol.FullyConnected(act1, name="fc4", num_hidden=384)
act4 <- mx.symbol.Activation(fc4, name="relu4", act_type="relu")

fc44 <- mx.symbol.FullyConnected(act4, name="fc44", num_hidden=2)
softmax <- mx.symbol.SoftmaxOutput(fc44, name="sm")

devices <- mx.cpu()
mx.set.seed(0)
newlist <- 
  model <- mx.model.FeedForward.create(softmax, X=train.x, y=train.y,
                                       ctx=devices, num.round=1000, array.batch.size=20,
                                       learning.rate=0.001, momentum=0.9,  eval.metric=mx.metric.accuracy,
                                       initializer=mx.init.uniform(0.05), 
                                       epoch.end.callback=mx.callback.log.train.metric(20))

??momentum
??optimizer
model$arg.params

# input
data <- mx.symbol.Variable('data')
# first conv
conv1 <- mx.symbol.Convolution(data=data, kernel=c(5,5), num_filter=20)
tanh1 <- mx.symbol.Activation(data=conv1, act_type="tanh")
pool1 <- mx.symbol.Pooling(data=tanh1, pool_type="max",
                           kernel=c(2,2), stride=c(2,2))
# second conv
conv2 <- mx.symbol.Convolution(data=pool1, kernel=c(5,5), num_filter=50)
tanh2 <- mx.symbol.Activation(data=conv2, act_type="tanh")
pool2 <- mx.symbol.Pooling(data=tanh2, pool_type="max",
                           kernel=c(2,2), stride=c(2,2))
# first fullc
flatten <- mx.symbol.Flatten(data=pool2)
fc1 <- mx.symbol.FullyConnected(data=flatten, num_hidden=500)
tanh3 <- mx.symbol.Activation(data=fc1, act_type="tanh")
# second fullc
fc2 <- mx.symbol.FullyConnected(data=tanh3, num_hidden=10)
# loss
lenet <- mx.symbol.SoftmaxOutput(data=fc2)

#the FFN model
mx.set.seed(0)
tic <- proc.time()
model <- mx.model.FeedForward.create(lenet, X=train.x, y=train.y,
                                     ctx=device.cpu, num.round=1, array.batch.size=100,
                                     learning.rate=0.05, momentum=0.9, wd=0.00001,
                                     eval.metric=mx.metric.accuracy,
                                     epoch.end.callback=mx.callback.log.train.metric(100))