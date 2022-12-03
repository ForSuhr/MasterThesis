# Generate random parameters
W = rand(2, 3)
b = rand(2)
# Perform the feedforward propagation
using NNlib
predict(x) = sigmoid((W * x) .+ b)
# Define the loss function
using Statistics
loss(x, y) = mean(abs2, ( predict(x) .- y))
# Compute the loss
input = rand(3, 100)
estimated_value = predict(input)
target_value = rand(2, 100)
loss(input, target_value)


# The ADAM algorithm with the learning rate α=0.01 and decay rates β1=0.9, β2=0.999.
using Flux
opt = Flux.Optimise.ADAM(0.01, (0.9, 0.999))
# Construct a function to compute the gradients
θ = Flux.params(W, b)
using Zygote
gs(x, y) = Zygote.gradient(() -> loss(x, y), θ)
# Construct a dataloader. "dataloader" is an iterable object, which yields a batch of data with the specified batchsize in each iteration. For instance, now we have 1000 points in the training set (x, y). A dataloader with the given batchsize 10 will generate only 10 points in each iteration.
dataloader = Flux.Data.DataLoader((input, target_value), batchsize = 10)

# Update the parameters with the given learning rate and optimization algorithm
for (x, y) in dataloader
    for θ in (W, b) 
        Flux.Optimise.update!(opt, θ, gs(x, y)[θ])
    end
    println("loss: ", loss(x, y))
end
