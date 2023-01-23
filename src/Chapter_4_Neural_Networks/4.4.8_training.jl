using Random # number generator
using NNlib: sigmoid # activation function
using Statistics: mean
using Zygote: gradient

# for reproducibility
rng = Random.default_rng()
Random.seed!(rng, 0)

# hand-written single-layer NN
predict(x, W, b) = sigmoid((W * x) .+ b)

# generate random initial parameters
W = rand(rng, 2, 3) # random 2 × 3 matrix
b = rand(rng, 2)    # random vector in ℝ²

# generate random training data
input = rand(rng, 3)
target = rand(rng, 2)

# mean squared error
loss(input, target, W, b) = mean(abs2, predict(input, W, b) .- target)

# value of loss function before training
loss₁ = loss(input, target, W, b)

# parameter update using gradient descent
α = 0.1 # learning rate
# loss as a function of parameters only
fwd = (W, b) -> loss(input, target, W, b)
gradW, gradb = gradient(fwd, W, b)
W = W - α * gradW
b = b - α * gradb
loss₂ = loss(input, target, W, b)

@assert loss₂ < loss₁ # training reduced the loss
"One update lead to a reduction of the error by $((loss₁ - loss₂) / loss₁ * 100) %"
