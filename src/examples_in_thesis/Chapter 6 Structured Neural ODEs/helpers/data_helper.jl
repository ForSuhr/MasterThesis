module DataHelper


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
    m, c, d, θ_0, θ_d, α = params
    v = p/m
    dz[1] = v
    dz[2] = -q/c-d*v
    dz[3] = d*((v)^2)/θ_d-α*(θ_d-θ_0)/θ_d
    dz[4] = α*(θ_d-θ_0)/θ_0
end


using OrdinaryDiffEq, SciMLBase, CommonSolve
function ODESolver(ODE_function, params, initial_state, time_span, time_step, numerical_method=ImplicitMidpoint())
    prob = ODEProblem(ODEFunction(ODE_function), initial_state, time_span, params)
    ode_data = Array(CommonSolve.solve(prob, numerical_method, tstops = time_step))
    return ode_data
end


using ReverseDiff, SciMLSensitivity
function NeuralODESolver(NeuralODE, θ, initial_state, time_span, time_step, numerical_method=ImplicitMidpoint())
    IVP = SciMLBase.ODEProblem(ODEFunction(NeuralODE), initial_state, time_span, θ)
    sensitivity_analysis = InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))
    neural_ode_data = CommonSolve.solve(IVP, numerical_method, p=θ, tstops = time_step, sensealg=sensitivity_analysis) 
    return neural_ode_data
end


using FiniteDiff
function SymplecticGradient(NN, ps, st, z)
  dH = FiniteDiff.finite_difference_gradient(x -> sum(NN(x, ps, st)[1]), z)
  return vec(cat(dH[2:2, :], -dH[1:1, :], dims=1))
end


export ODEfunc_udho, ODEfunc_idho, ODEfunc_ndho, ODESolver, NeuralODESolver, SymplecticGradient


end
