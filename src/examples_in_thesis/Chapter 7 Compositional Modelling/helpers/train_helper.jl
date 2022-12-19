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


function SolveIVP(NeuralODE, θ, initial_state, batch_timesteps, numerical_method=ImplicitMidpoint(), implicit=true, sensitivity_analysis=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)))
    IVP = SciMLBase.ODEProblem(ODEFunction(NeuralODE), initial_state, (batch_timesteps[1], batch_timesteps[end]), θ)
    if implicit
        pred_data = Array(CommonSolve.solve(IVP, numerical_method, p=θ, tstops = batch_timesteps, sensealg=sensitivity_analysis))
    else
        pred_data = Array(CommonSolve.solve(IVP, numerical_method, p=θ, saveat = batch_timesteps, sensealg=sensitivity_analysis))
    end
    return pred_data
end


function OptFunction(loss_function, adtype=Optimization.AutoZygote())
    optimization_function = Optimization.OptimizationFunction((θ, ps, batch_data, batch_timesteps) -> loss_function(θ, batch_data, batch_timesteps), adtype)
    return optimization_function
end


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


export SolveIVP, OptFunction, FluxTrain, LuxTrain

end