using Pkg
Pkg.precompile()
Pkg.add(".jl")
Pkg.rm("Atom.jl")
