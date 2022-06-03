# Report

## Physical Model

1. undamped harmonic oscillator (Hamitlonian system)

2. isothermal damped harmonic oscillator (EPHS)

3. nonisothermal damped harmonic oscillator (EPHS)

- explain the models and their physical meaning
    - in particular, use bond-graph diagrams/composition patterns
- generate train and test data


## Neural ODE (baseline)

baseline -> Neural ODE makes basically no assumptions about the actual physical model

dx/dt = RHS(x) where RHS is a NN

apply it 1., 2., 3.

- explain how the approach works
    - NN architecture
    - number of parameters
- time the training process
- show results and comment about your observations
    - accuracy (training data, test data)
- problems / open questions


this approach is pretty much 100% neural network-based
problems:
- needs a lot of training data
- lacks interpretability (black-box model)



## Structured Neural ODE

1. Hamiltonian NN

assume that
x = [q, p]
makes sense for mechanical systems whose state is
configuration q and conjugate momenta p

also assume that there is essentially no friction

then you have a Hamiltonian system of canonical form
RHS is
dq/dt = +dH/dp
qp/dt = -dH/dq   (written as a matrix: +1 and -1 on off diagonal)
where H is a NN


                mass       + spring  (2 blue components)
for 1. H(q, p) = p^2/(2*m) + q^2/(2*c) 


2. and 3. (adapted) SymODEN

generalization of the above
because the assumptions are less strict

- general Hamiltonian structure (rather than canonical Hamiltonian structure)
  (general skew-symmetric matrix)
- there may be friction/dissipation

...


- the approach is supposed to be more data-efficient (less data needed to obtain an accurate model)
- the resulting model has the assumed structure but it is still monolithic


## EPHS with unknown parameters

1. unknown mass m, spring compliance c (1/stiffness)
2. also unknown damping coefficient d
3. also unknown heat transfer coefficient α

this is the extreme case where we assume complete knowledge of the evolution equations
except for some unknown (scalar) parameters (which have direct physical meaning)



## outlook

combine the above approach with small neural networks (or other approximators):
for example replace the spring term by a neural network (since the spring might not be linear)
we only have a NN with one input and one output 
H_{spring} = NN(q)
H_{spring} = q^2/(2 NN(q))  (probably unnecessarily complicated, but allows for varying stiffness interpretation of NN output)
