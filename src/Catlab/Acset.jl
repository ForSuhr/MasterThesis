# Acset
## using package
using Catlab, Catlab.Theories
using Catlab.CategoricalAlgebra
using Catlab.WiringDiagrams
using Catlab.Graphics
using Catlab.Graphics.Graphviz
using Catlab.Programs
using Catlab.WiringDiagrams
import Catlab.Graphics: Graphviz

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

## check the elements in C
generators(C)

## illustrate the free category C
to_graphviz(C)

## define Acset
@acset_type Acset(C, index=[:box, :ij, :bj, :ipt, :jpt, :bpt]) <: WiringDiagrams.DirectedWiringDiagrams.AbstractWiringDiagram

## implement an instance
    md = @acset MyAcset begin
        Box = 2
        Port = 3
        BoundaryPort = 1
        Junction = 2
        PortType = 2
        box = [1,2,2]
        ij = [1,1,2]
        bj = [2]
        ipt = [1,1,2]
        jpt = [1,2]
        bpt = [2]
    end



