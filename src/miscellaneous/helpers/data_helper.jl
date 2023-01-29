module DataHelper
using OrdinaryDiffEq, SciMLBase, CommonSolve
using ReverseDiff, SciMLSensitivity
using FiniteDiff


function ODEfunc_udho(dz, z, params, t)
    q, p = z
    m, c = params
    v = p/m
    dz[1] = v
    dz[2] = -q/c
end


function ODEfunc_idho(dz, z, params, t)
    q, p, s_e = z
    m, c, d, θ_0 = params
    v = p/m
    dz[1] = v
    dz[2] = -q/c -d*v
    dz[3] = d*v^2/θ_0
end


function ODEfunc_ndho(dz,z,params,t)
    q, p, s_e, s_d = z
    m, c, d, θ_0, α, c_tc = params
    θ_d = exp(s_d/c_tc) / c_tc
    Δθ = θ_d - θ_0
    v = p/m
    dz[1] = v
    dz[2] = -q/c-d*v
    dz[3] = α*(Δθ)/θ_0
    dz[4] = d*((v)^2)/θ_d-α*(Δθ)/θ_d
end


function ODESolver(ODE_function, params, initial_state, time_span, time_step, numerical_method=ImplicitMidpoint(), implicit=true)
    prob = ODEProblem(ODEFunction(ODE_function), initial_state, time_span, params)
    if implicit
        ode_data = Array(CommonSolve.solve(prob, numerical_method, tstops = time_step))
    else
        ode_data = Array(CommonSolve.solve(prob, numerical_method, saveat = time_step))
    end
    return ode_data
end


function NeuralODESolver(NeuralODE, θ, initial_state, time_span, time_step, numerical_method=ImplicitMidpoint(), implicit=true)
    IVP = SciMLBase.ODEProblem(ODEFunction(NeuralODE), initial_state, time_span, θ)
    sensitivity_analysis = InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))
    if implicit
        neural_ode_data = CommonSolve.solve(IVP, numerical_method, p=θ, tstops=time_step, sensealg=sensitivity_analysis) 
    else
        neural_ode_data = CommonSolve.solve(IVP, numerical_method, p=θ, saveat=time_step, sensealg=sensitivity_analysis) 
    end
    return neural_ode_data
end


function SymplecticGradient(NN, ps, st, z)
  ∂H = FiniteDiff.finite_difference_gradient(x -> sum(NN(x, ps, st)[1]), z)
  return vec(cat(∂H[2:2, :], -∂H[1:1, :], dims=1))
end


export ODEfunc_udho, ODEfunc_idho, ODEfunc_ndho, ODESolver, NeuralODESolver, SymplecticGradient


end
