# isothermal damped harmonic oscillator UndirectedWiringDiagrams
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

## define the free category C
@present C(FreeSchema) begin
    (Box,Port,Junction,BoundaryPort,PortType)::Ob
    box::Hom(Port,Box)
    ij::Hom(Port,Junction)
    bj::Hom(BoundaryPort, Junction)
    ipt::Hom(Port,PortType)
    jpt::Hom(Junction,PortType)
    bpt::Hom(BoundaryPort,PortType)
end

## illustrate the free category C
to_graphviz(C)

## check the elements in C
generators(C)

## define Acset
@acset_type Acset(C, index=[:box, :ij, :bj, :ipt, :jpt, :bpt]) <: WiringDiagrams.DirectedWiringDiagrams.AbstractWiringDiagram

## implement an instance
idho = @acset Acset begin
  Box = 5
  Port = 7
  BoundaryPort = 0
  Junction = 3
  PortType = 3
  box = [1,2,2,3,4,4,5]
  ij = [1,1,2,2,2,3,3]
  bj = []
  ipt = [1,1,2,2,2,3,3]
  jpt = [1,2,3]
  bpt = []
end

add_part!(idho,:Box)
idho
add_part!(idho,:Port,box=6,ipt=1,ij=1)
idho
subpart(idho,3,:jpt)
idho

## UWD for isothermal damped harmonic oscillator
uwd = @relation () begin
  spring(j₁)
  Dₘ(j₁,j₂)
  mass(j₂)
  damping(j₂,j₃)
  environment(j₃)
end

## illustrate the UWD
draw(uwd)

# UndirectedWiringDiagrams.add_box!
UndirectedWiringDiagrams.add_box!(uwd,)
draw(uwd)




