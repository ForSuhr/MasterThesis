using Zygote
## gradient of the neural network
dNN_dq = Zygote.gradient(u -> NN(u, initial_params(NN))[1], u0[1])[1] ## ∂H/∂q
dNN_dp = Zygote.gradient(u -> NN(u, initial_params(NN))[2], u0[2])[1] ## ∂H/∂p