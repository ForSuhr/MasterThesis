using ForwardDiff

function Hamiltonian_idho(m,c)
    H(q,p) = p^2/2m + q^2/2c ### H = T + V, where T is kinetic energy; V is potential energy; p is the momentum of the mass; m is the mass; q is the displacement of the spring; c is the spring compliance
    ∂H_∂q(q) = ForwardDiff.derivative(q -> H(q,0), q) ### the partial derivative of H with respect to q
    ∂H_∂p(p) = ForwardDiff.derivative(p -> H(0,p), p) ### the partial derivative of H with respect to p
    return ∂H_∂q, ∂H_∂p
  end

Hamiltonian_idho(1,1)[1](4) ### (1,1) means (m,c); [1](5) means ∂H_∂q(5)
Hamiltonian_idho(1,1)[2](5) ### (1,1) means (m,c); [2](4) means ∂H_∂q(4)

I = [0 1; -1 0]
D = [0 0; 0 d]
dd = (I-D)*Hamiltonian_idho(1,1)[1]