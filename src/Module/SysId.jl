module SysId
using Plots


include("ODE/integrator.jl")
include("ODE/training.jl")


export timesteps, ODE_integrator, Tsit5
export neural_network, params_initial, neural_ODE_integrator, decompose_ps, decompose_st
export train, plot, plot!

end