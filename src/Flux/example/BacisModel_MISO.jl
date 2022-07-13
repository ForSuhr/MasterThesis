# https://fluxml.ai/Flux.jl/stable/models/basics/
using Revise
using Flux
# do simulation
actual(x1, x2) = 10x1+2x2+4
x1_train = hcat(0:5...)
x2_train = hcat(6:11...)
y_train = actual.(x1_train, x2_train)
x1_test = hcat(11:15...)
x2_test = hcat(16:20...)
y_test = actual.(x1_test, x2_test)

# build a predict model with miso
predict = Chain(
  Dense(2 => 20, tanh),
  Dense(20 => 5, tanh),
  Dense(5 => 1))

x_train = vcat(x1_train,x2_train)
predict(x_train)
x_test = vcat(x1_test,x2_test)

# build a loss function mean squared error to tell Flux how to evaluate the quality of a prediction
loss(x,y) = Flux.Losses.mse(predict(x_train),y)
loss(x_train,y_train)

# improve the prediction with the optimiser Descent()
using Flux:train!
opt = Descent(0.01)
data = [(x_train, y_train)]
# data = [(x_train, y_train),(x_test, y_test)]

# use a function to collect weight and bias
predict.weight
predict.bias
parameters = Flux.params(predict)
predict.weight in parameters, predict.bias in parameters

# train the model with the gradient descent optimiser
train!(loss, parameters, data, opt)

#check the loss, the ideal loss should be close to zero; check the parameters weight and bias
loss(x_train, y_train)
parameters

# Iteratively train the model
for epoch in 1:1000
    train!(loss, parameters, data, opt)
end

#check the loss, the ideal loss should be close to zero; check the parameters weight and bias
loss(x_train, y_train)
parameters

# Verify the results
predict(x_test)
y_test
