# Generate random parameters
W = rand(2, 3)
b = rand(2)
# Perform the feedforward propagation
using NNlib: sigmoid
predict(x) = sigmoid((W * x) .+ b)
# Define the loss function
using Statistics: mean
loss(x, y) = mean(abs2, ( predict(x) .- y))


# Compute the loss
input = rand(3)
estimated_value = predict(input)
target_value = rand(2)
loss(input, target_value)


# "Flux" is a deep learning framework in Julia language. The function "params" creates a trainable parameters object.
using Flux
θ = Flux.params(W, b)
# Compute the gradient of the loss. "Zygote" is an automatic differentiation package.
using Zygote
gs = Zygote.gradient(() -> loss(input, target_value), θ)
# Compute the gradient of the loss with respect to the parameters
gs[W]
gs[b]


# Single update of the parameters accroading to gradient descent 
# The learning Rate
α = 0.1
# Update the parameters using gradient descent
for θ in (W, b)
    θ .-= α * gs[θ]
end
loss(input, target_value)


# The optimization algorithm "Gradient Descent"
opt = Flux.Descent(0.1)
# Update the parameters with the given learning rate and optimization algorithm
for θ in (W, b)
    Flux.Optimise.update!(opt, θ, gs[θ])
end
loss(input, target_value)
