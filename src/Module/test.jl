include("SysId.jl")
using Main.SysId


function ODEfunc_udho(du,u,params,t)
    ## conversion
    q, p = u
    m, c = params
    ## ODEs
    du[1] = p/m
    du[2] = -q/c
end


u0 = [1.0, 1.0]
ODE_params = [1.5, 1.0]
tsteps = timesteps(0.0, 0.2, 20.0); # start, step, stop


# ground truth
trajectories_ground_truth = ODE_integrator(ODEfunc_udho, u0, tsteps, ODE_params, Tsit5())


# prediction
NN = neural_network(2, 20, 2)
neural_params = params_initial(NN);
trajectories_prediction = neural_ODE_integrator(NN, u0, tsteps, decompose_ps(neural_params), decompose_st(neural_params), Tsit5())


# loss function
function loss_neuralode(params)
    trajectories_prediction = neural_ODE_integrator(NN, u0, tsteps, params, decompose_st(neural_params), Tsit5())
    loss = sum(abs2, trajectories_ground_truth .- trajectories_prediction)
    return loss, trajectories_prediction
end


# callback
callback = function(params, loss, pred_data)
    ### plot Ground truth and prediction data
    println(loss)
    x_axis_ode_data = trajectories_ground_truth[1,:]
    y_axis_ode_data = trajectories_ground_truth[2,:]
    x_axis_pred_data = pred_data[1,:]
    y_axis_pred_data = pred_data[2,:]
    plt = plot(x_axis_ode_data, y_axis_ode_data, label="Ground truth", xlabel="q", ylabel="p")
    plot!(plt,x_axis_pred_data, y_axis_pred_data, label = "Prediction", xlabel="q", ylabel="p")
    """
    q = pred_data[1,:]
    p = pred_data[2,:]
    m, c = init_params
    H = p.^2/(2m) + q.^2/(2c)
    plt = plot(tsteps, q, label="Position")
    plt = plot!(tsteps, p, label="Momentum")
    plt = plot!(tsteps, H, label="Hamiltonian")
    """
    display(plot(plt))
    if loss > 0.1 
      return false
    else
      return true
    end
  end


# training
result = train(loss_neuralode, decompose_ps(neural_params), callback, learning_rate=0.05, maxiters=100)
result = train(loss_neuralode, result.u, callback, learning_rate=0.01, maxiters=100)
result = train(loss_neuralode, result.u, callback, learning_rate=0.001, maxiters=300)
