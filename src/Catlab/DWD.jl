# isothermal damped harmonic oscillator DirectedWiringDiagrams
## using package
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

draw(uwd::AbstractUWD) = to_graphviz(uwd, junction_labels=:variable, box_labels=:name)

## define a free biproduct category (objects and morphisms)
A, B, C, D = Ob(FreeBiproductCategory, :A, :B, :C, :D)
f = Hom(:f, A, B)
g = Hom(:g, B, C)
h = Hom(:h, C, D)

## convert morphisms into a diagram with a single box 
f, g, h = to_wiring_diagram(f), to_wiring_diagram(g), to_wiring_diagram(h)
draw(f)

## try some compositions
compose(f,g)
draw(compose(f,g))

## manipulate wiring diagrams by using imperative interface
f = Box(:f, [:A], [:B])
g = Box(:g, [:B], [:C])
h = Box(:h, [:C], [:D])

## manually construct a composition of two boxes
d = WiringDiagram([:A], [:C])
draw(d)
fv = add_box!(d, f)
draw(d)
gv = add_box!(d, g)
draw(d)
add_wires!(d, [
  (input_id(d),1) => (fv,1),
  (fv,1) => (gv,1),
  (gv,1) => (output_id(d),1),
])
draw(d)
### query the number of boxes and wires
nboxes(d)
nwires(d)