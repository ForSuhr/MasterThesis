# 3 neurons in this layer and 2 neurons in the next layer, generate random parameters
W1 = rand(2, 3)
b1 = rand(2)
# 2 neurons in this layer and 4 neurons in the next layer, generate random parameters
W2 = rand(4, 2)
b2 = rand(4)

# Input
a1 = rand(3)
# The first layer
z2 = (W1 * a1) .+ b1
using NNlib
a2 = sigmoid(z2)
# The second layer
z3 = (W2 * a2) .+ b2
a3 = sigmoid(z3)