using Random
using NNlib: sigmoid
using Statistics: mean
using Zygote: gradient

rng = Random.default_rng()
Random.seed!(rng, 0)

# hand-written single-layer NN
predict(x, W, b) = sigmoid((W * x) .+ b)

# generate random initial parameters
W = rand(rng, 2, 3)
b = rand(rng, 2)

# generate random training data
input = rand(rng, 3)
target = rand(rng, 2)

# mean squared error
loss(input, target, W, b) = mean(abs2, predict(input, W, b) .- target)

# value of loss function before training
loss₁ = loss(input, target, W, b)

# parameter update via gradient descent
α = 0.1 # learning rate
fwd = (W, b) -> loss(input, target, W, b)
gradW, gradb = gradient(fwd, W, b)
W = W - α * gradW
b = b - α * gradb
loss₂ = loss(input, target, ps)

@assert loss₂ < loss₁
"One update lead to a reduction of the error by $((loss₁ - loss₂) / loss₁ * 100) %"
