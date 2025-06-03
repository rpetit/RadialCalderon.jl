# # Evaluating the forward map and its derivatives

#=
Here we show how to evaluate the forward map associated to the Calderon problem with radial piecewise constant conductivities. We also show how to evaluate its derivatives via automatic differentiation.
=#

using RadialCalderon

# ## Setting

#=
We define the forward problem. There are three annuli with the inner radii being 0.5 and 0.25.
=#

n = 3
r = [0.5, 0.25]
forward = ForwardProblem(r);

# ## Forward map

#=
We define the (diagonal) Neumann-to-Dirichlet map associated to the boundary data $(\mathrm{cos}(j\theta))_{1\leq j\leq m}$.
=#

m = 5
λ(j, σ) = forward_map(forward, j, σ)
Λ(σ) = [λ(j, σ) for j in 1:m];

# ## Derivatives

#=
The first and second order derivatives of $\Lambda$ can be computed via automatic differentiation. The automatic differentiation backend (here [ForwardDiff](https://github.com/JuliaDiff/ForwardDiff.jl)) can be changed easily thanks to the [DifferentiationInterface](https://github.com/JuliaDiff/DifferentiationInterface.jl) package.
=#

using DifferentiationInterface
import ForwardDiff

backend = AutoForwardDiff()

dΛ(σ) = jacobian(Λ, backend, σ)  # Jacobian of the forward map
grad_λ(j, σ) = gradient(σ -> λ(j, σ), backend, σ)  # gradient of the j-th output
hess_λ(j, σ) = hessian(σ -> λ(j, σ), backend, σ)  # Hessian of the j-th output

dΛ(ones(n))

# ## Closed form expression

#=
When $n=3$ and the conductivity is equal to $1$ on the outermost annulus, the forward map has the following closed form expression [harrachCalderonProblemFinitely2023](@cite):

$λ_j(σ_1,σ_2)=\frac{c_j+d_j}{j(c_j-d_j)},$

where

$c_j=(1/σ_1+1)(σ_1+σ_2) + (1/σ_1-1)(σ_1-σ_2)r_2^{2j}/r_1^{2j},$
$d_j=(1/σ_1-1)(σ_1+σ_2)r_1^{2j} + (1/σ_1+1)(σ_1-σ_2)r_2^{2j}.$

We check the consistency of the implemented forward map with this formula.
=#

σ1 = 1.5
σ2 = 0.5
σ = [1.0, σ1, σ2]

c(j) = (1/σ1+1)*(σ1+σ2) + (1/σ1-1)*(σ1-σ2)*r[2]^(2*j)/r[1]^(2*j)
d(j) = (1/σ1-1)*(σ1+σ2)*r[1]^(2*j) + (1/σ1+1)*(σ1-σ2)*r[2]^(2*j)

@info forward_map(forward, 1, σ) ≈ (c(1)+d(1)) / (c(1)-d(1))
@info forward_map(forward, 2, σ) ≈ (c(2)+d(2)) / (2*(c(2)-d(2)))
@info forward_map(forward, 3, σ) ≈ (c(3)+d(3)) / (3*(c(3)-d(3)))

# ## References

#=
```@bibliography
Pages = ["forward.md"]
Canonical = false
```
=#