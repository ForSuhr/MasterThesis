- learn some basic Julia
- study machine learning
    - study some Flux.jl examples
    - in particular focus on neural ODEs
    - try to develop some (port-)Hamiltonian neural ODEs
        - Hamiltonian Neural Networks
        - Dissipative SymODEN
- EPHS.jl software library
    1. UWD syntax
        - study acsets (theory, code, and exmples)
            - theroy: https://arxiv.org/abs/2106.04703
            - code: src/categorical_algebra/ACSetInterface.jl
            - examples: ...
        - study src/wiring_diagrams/Undirected.jl
        - maybe (later) we should implement UWD
          according to paper and our own needs incl. port types
        - create some example systems (consider examples from articles)
    2. EPHS semantics ("associate differential-algebraic equations to the syntax")
        - ModelingToolkit.jl
    3. simulation
        - DifferentialEquations.jl
    4. system identification based on machine learning
        - Hamiltonian Neural Networks
        - Dissipative SymODEN
