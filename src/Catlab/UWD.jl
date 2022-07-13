# UWD
## using package
using Revise
using Catlab, Catlab.Theories
using Catlab.CategoricalAlgebra
using Catlab.WiringDiagrams
using Catlab.Graphics
using Catlab.Graphics.Graphviz
using Catlab.Programs
using Catlab.WiringDiagrams

## define a function draw() for graphviz
draw(d::WiringDiagram) = to_graphviz(d,
  orientation=LeftToRight,
  labels=true, label_attr=:xlabel,
  node_attrs=Graphviz.Attributes(
    :fontname => "Courier",
  ),
  edge_attrs=Graphviz.Attributes(
    :fontname => "Courier",
  )
)

## draw out TheoryUWD that already defined in Catlab.jl
to_graphviz(WiringDiagrams.UndirectedWiringDiagrams.TheoryUWD)

## Construct an UWD using relation notation.
uwd = @relation (x, y, z) begin
  R(x,y)
  S(y,z)
end

## then illustrate those relations
draw(uwd)

## another example for some other relations. 
## Note that y was not seen as an outer_junction because y is not written in @relation (x, z)
uwd₂ = @relation (x, z) begin
  R(x,y)
  S(y,z)
  T(x,y,z)
end

draw(uwd₂)
