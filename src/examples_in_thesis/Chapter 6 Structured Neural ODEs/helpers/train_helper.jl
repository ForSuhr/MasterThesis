#################################
# This is a module for training #
#################################

module TrainInterface
using Optimization
using OptimizationOptimisers
using IterTools: ncycle
using Lux: ComponentArray
using ReverseDiff, SciMLSensitivity
using OrdinaryDiffEq, SciMLBase, CommonSolve


function SolveIVP(NeuralODE, θ, initial_state, batch_timesteps, numerical_method=ImplicitMidpoint(), sensitivity_analysis=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)))
    IVP = SciMLBase.ODEProblem(ODEFunction(NeuralODE), initial_state, (batch_timesteps[1], batch_timesteps[end]), θ)
    pred_data = Array(CommonSolve.solve(IVP, numerical_method, p=θ, tstops = batch_timesteps, sensealg=sensitivity_analysis))
    return pred_data
end


function OptFunction(loss_function, adtype=Optimization.AutoZygote())
    optimization_function = Optimization.OptimizationFunction((θ, ps, batch_data, batch_timesteps) -> loss_function(θ, batch_data, batch_timesteps), adtype)
    return optimization_function
end


function FluxTrain(optimization_function, θ, α, epochs, dataloader, callback)
    optprob = Optimization.OptimizationProblem(optimization_function, θ)
    result = Optimization.solve(optprob, Optimisers.ADAM(α), ncycle(dataloader, epochs), callback=callback)
    θ = result.u
    return θ
end


function LuxTrain(optimization_function, θ, α, epochs, dataloader, callback)
    optprob = Optimization.OptimizationProblem(optimization_function, ComponentArray(θ))
    result = Optimization.solve(optprob, Optimisers.ADAM(α), ncycle(dataloader, epochs), callback=callback)
    θ = result.u
    return θ
end


export SolveIVP, OptFunction, FluxTrain, LuxTrain

end
