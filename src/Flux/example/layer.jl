using Revise
using Flux
f(x, y) = sum((x .- y).^2)
# partial derivative
gradient(f,[2, 1], [2, 0])

# work with collections of parameters via the params functions
x = [2, 1];
y = [2, 0];
gs = gradient(Flux.params(x, y)) do
         f(x, y)
       end
gs[x]
gs[y]

# build a model
W = rand(2, 5)
b = rand(2)
predict(x) = W*x .+ b

# define loss function
function loss(x, y)
  ŷ = predict(x)
  sum((y .- ŷ).^2)
end

# use some dummy data to show the loss value
x, y = rand(5), rand(2) # Dummy data
loss(x, y) # ~ 3

# do the same thing in train!(), it is a single step to train the model
gs = gradient(() -> loss(x, y), Flux.params(W, b))

# do the same thing in optimiser
W̄ = gs[W]
W .-= 0.1 .* W̄

# check the loss
loss(x, y) # ~ 2.5

"""
# build layers
## layer1
W1 = rand(3, 5)
b1 = rand(3)
layer1(x) = W1 * x .+ b1
##layer2
W2 = rand(2, 3)
b2 = rand(2)
layer2(x) = W2 * x .+ b2
## combine the layers
model(x) = layer2(σ.(layer1(x)))
## test the combined model
model(rand(5))
"""

# a better way to build layers, which is the same thing we did in Dense()
function layer(in, out)
    W = randn(out, in)
    b = randn(out)
    x -> W * x .+ b
  end
  
  layer1 = layer(5, 3) # we can access layer1.W etc
  layer2 = layer(3, 2)
  layer1.W
  
  model(x) = layer2(σ.(layer1(x)))
  
  model(rand(5))


# we can just use Dense()  
model = Chain(
  Dense(10 => 5, σ),
  Dense(5 => 2),
  softmax)

model(rand(10))

# or just composite the functions
m = Dense(5 => 2) ∘ Dense(10 => 5, σ)
rand(10)
m(rand(10))

