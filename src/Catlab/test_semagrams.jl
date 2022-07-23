using Pkg
Pkg.build("PyCall")
using Semagrams.Examples

p = Semagram{ReactionNet{Float64, String}}(ReactionNetSema)

# Edit semagram (see docs)

get_acset(p)