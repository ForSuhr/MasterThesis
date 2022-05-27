using Revise
using Flux
# do simulation
actual(x) = 8x + 4
x_train, x_test= hcat(0:5...), hcat(6:10...)
y_train, y_test = actual.(x_train), actual.(x_test)

# build a predict model with siso, the single input is x_train and the single output is y_train
predict = Dense(1 => 1)
predict(x_train)

# build a loss function mean squared error to tell Flux how to evaluate the quality of a prediction
loss(x,y) = Flux.Losses.mse(predict(x),y)
loss(x_train,y_train)

# improve the prediction with the optimiser Descent()
using Flux:train!
opt = Descent(0.1)
data = [(x_train, y_train)]
# data = [(x_train, y_train),(x_test, y_test)]

# use a function to collect weight and bias
predict.weight
predict.bias
parameters = Flux.params(predict)
predict.weight in parameters, predict.bias in parameters

# train the model with the gradient descent optimiser
train!(loss, parameters, data, opt)

# check the loss, the ideal loss should be close to zero; check the parameters weight and bias
loss(x_train, y_train)
parameters

# creat a void array
array = Vector{Float64}(undef,1000)

# Iteratively train the model
for epoch in 1:1000
    train!(loss, parameters, data, opt)
    array[epoch] = loss(x_train, y_train)
end

# plot the iteration of loss 
using Plots
plot(1:1000, array)

#check the loss, the ideal loss should be close to zero; check the parameters weight and bias
loss(x_train, y_train)
parameters

# Verify the results
predict(x_test)
y_test
