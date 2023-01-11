using Random
using NNlib: sigmoid
using Lux
using Statistics: mean
using Optimisers: Adam
using Zygote: gradient

rng = Random.default_rng()
Random.seed!(rng, 0)

model = Dense(3 => 2, sigmoid)

# generate random training data
data = (
    x = rand(rng, 3),
    y = rand(rng, 2)
)

# mean squared error loss function matching the Lux.Training API
function loss_function(model, ps, st, data)
    y_pred, st = Lux.apply(model, data.x, ps, st)
    mse_loss = mean(abs2, y_pred .- data.y)
    return mse_loss, st, ()
end

# setup Adam optimizer
α = 0.01 # learning rate
β₁, β₂ = 0.9, 0.999  # decay of momenta
opt = Adam(α, (β₁, β₂))

tstate = Lux.Training.TrainState(rng, model, opt)
vjp_rule = Lux.Training.ZygoteVJP() # use reverse-mode AD

function train!(tstate, vjp, data, epochs)
    for epoch in 1:epochs
        grads, loss, states, tstate =
            Lux.Training.compute_gradients(
                vjp,
                loss_function,
                data,
                tstate
            )
        @info epoch=epoch loss=loss
        tstate = Lux.Training.apply_gradients(tstate, grads)
    end
    return tstate
end

loss₁ = loss_function(tstate.model, tstate.parameters, tstate.states, data)[1]

tstate = train!(tstate, vjp_rule, data, 1)

loss₂ = loss_function(tstate.model, tstate.parameters, tstate.states, data)[1]

@assert loss₂ < loss₁
"One update lead to a reduction of the error by $((loss₁ - loss₂) / loss₁ * 100) %"
