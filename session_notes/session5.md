First-order explicit ODE
dx/dt = f(x)

Many isolated PHS can be written in this form (explicit isolated PHS)
dx/dt = f(x) = (J(x) - R(x)) dH/dx

If there are constraints then you will get a DAE (implicit isolated PHS)
F(x) dx/dt = (J(x) - R(x)) dH/dx
for F(x) = I (identity matrix) this again becomes an ODE
Constraints in the system correspond to F(x) not having full rank (for all x).
For example if you have capacitors in parallel.

Neural ODE
Usually one wants to learn a first-order explicit ODE
dx/dt = f(x)
given state trajectories
by taking f as a neural network:
f : current state -> change of the state
you parametrize f by a feed-forward neural network

Hence a neural ODE is a special case of
parameter estimation of ODEs
where the parameters are the weights of a feed-forward neural network.

```julia

function fiip(du,u,p,t)
  du[1] = dx = p[1]*u[1] - p[2]*u[1]*u[2]
  du[2] = dy = -p[3]*u[2] + p[4]*u[1]*u[2]
end

```

