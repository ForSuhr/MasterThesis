


using Plots
using StatsFuns
using DifferentialEquations

plot(logistic, xlabel="z", ylabel="σ(z)" , label="logistic sigmoid", framestyle=:origin, ylims=(0, 1.2))


plot(tanh, xlabel="z", ylabel="f(z)" , label="tanh", framestyle=:origin, ylims=(-1.4, 1.4))


plot(relu, xlabel="z", ylabel="ReLU(z)" , label="ReLU", legend=:topleft, framestyle=:origin)
