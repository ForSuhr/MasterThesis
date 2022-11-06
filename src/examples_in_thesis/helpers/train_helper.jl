#################################
# This is a module for training #
#################################

module TrainInterface
using Optimization
using OptimizationOptimisers
using IterTools: ncycle
using Lux: ComponentArray

function FluxTrain(optf, θ, α, epochs, dataloader, callback)
    optprob = Optimization.OptimizationProblem(optf, θ)
    result = Optimization.solve(optprob, Optimisers.ADAM(α), ncycle(dataloader, epochs), callback=callback)
    θ = result.u
    return θ
end

function LuxTrain(optf, θ, α, epochs, dataloader, callback)
    optprob = Optimization.OptimizationProblem(optf, ComponentArray(θ))
    result = Optimization.solve(optprob, Optimisers.ADAM(α), ncycle(dataloader, epochs), callback=callback)
    θ = result.u
    return θ
end

export FluxTrain, LuxTrain

end