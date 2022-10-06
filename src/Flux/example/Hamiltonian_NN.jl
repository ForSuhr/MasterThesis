# https://diffeqflux.sciml.ai/dev/examples/hamiltonian_nn/
## using package
using Revise
using DiffEqFlux, DifferentialEquations, Statistics, Plots
using ReverseDiff

## Data Generation: The HNN predicts the gradients ((˙q),(˙p))
## (\dot(q), \dot(p))((˙​q),(˙​p)) given (q,p)(q, p)(q,p). 
## Hence, we generate the pairs (q,p)(q, p)(q,p) using the equations given at the top. Additionally to supervise the training we also generate the gradients. Next we use use Flux DataLoader for automatically batching our dataset.
t = range(0.0f0, 1.0f0, length = 1024)
π_32 = Float32(π)
q_t = reshape(sin.(2π_32 * t), 1, :)
p_t = reshape(cos.(2π_32 * t), 1, :)
dqdt = 2π_32 .* p_t
dpdt = -2π_32 .* q_t

data = cat(q_t, p_t, dims = 1)
target = cat(dqdt, dpdt, dims = 1)
dataloader = Flux.Data.DataLoader((data, target); batchsize=256, shuffle=true)

## Training the HamiltonianNN: We parameterize the HamiltonianNN with a small MultiLayered Perceptron 
## (HNN also works with the Fast* Layers provided in DiffEqFlux). 
## HNNs are trained by optimizing the gradients of the Neural Network. 
## Zygote currently doesn't support nesting itself, so we will be using 
## ReverseDiff in the training loop to compute the gradients of the HNN Layer for Optimization.
hnn = HamiltonianNN(
    Chain(Dense(2, 64, relu), Dense(64, 1))
)

p = hnn.p

opt = ADAM(0.01) ### ADAM optimiser: Adaptive Moment Estimation 

loss(x, y, p) = mean((hnn(x, p) .- y) .^ 2)

callback() = println("Loss Neural Hamiltonian DE = $(loss(data, target, p))")

epochs = 100
for epoch in 1:epochs
    for (x, y) in dataloader
        gs = ReverseDiff.gradient(p -> loss(x, y, p), p) ## https://diffeqflux.sciml.ai/stable/layers/HamiltonianNN/#Hamiltonian-Neural-Network
        Flux.Optimise.update!(opt, p, gs)
    end
    if epoch % 100 == 1
        callback()
    end
end
callback()
Flux.train!
## Solving the ODE using trained HNN: In order to visualize the learned trajectories, we need to solve the ODE. We will use the NeuralHamiltonianDE layer which is essentially a wrapper over HamiltonianNN layer and solves the ODE.
model = NeuralHamiltonianDE(
    hnn, (0.0f0, 1.0f0),
    Tsit5(), save_everystep = false,
    save_start = true, saveat = t
)


pred = Array(model(data[:, 1]))
plot(data[1, :], data[2, :], lw=4, label="Original")
plot!(pred[1, :], pred[2, :], lw=4, label="Predicted")
xlabel!("Position (q)")
ylabel!("Momentum (p)")