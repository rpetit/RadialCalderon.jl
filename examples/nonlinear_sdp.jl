# # Reconstruction via convex programming

#=
In this tutorial, we show how to implement in practice the reconstruction method introduced in [harrachCalderonProblemFinitely2023](@cite). It is based on the resolution of a convex nonlinear semidefinite program of the form 

$\underset{\sigma\in[a,b]^n}{\mathrm{min}}~\langle c,\sigma\rangle~~\mathrm{s.t.}~~\Lambda(\sigma)\leq y,$

where $c$ is a weight vector with positive entries and $y$ is the vector of observations. If the number of measurements $m$ is large enough, then there exists a vector $c$ such that, for every $\sigma^\dagger\in[a,b]^n$, the above problem with ${y=\Lambda(\sigma^\dagger)}$ has a unique solution which is $\sigma^\dagger$.
=#

# ## Setting

#=
We define the forward problem. There are two annuli with the inner radius being 0.5.
=#

using RadialCalderon

n = 2
r = [0.5]
forward = ForwardProblem(r)

a = 0.5
b = 1.5;

# ## Estimation of the weight vector

#=
For a given $m$, one can rely on the function [`estimate_c`](@ref) to try to find a universal vector $c$. Below, we check that, when $n=2$, this is possible for $m=3$ but not for $m=2$. First, we define a set of conductivities which will be used to estimate $c$.
=#

using Base.Iterators: product

k = 2

prod_it = product([range(a, b, k) for i=1:forward.n]...)
σ = hcat(collect.(collect(prod_it))...)
σ = σ[:, 1:end-1];  # remove b*ones(n)

#=
We check that there is no admissible vector $c$ for $m=2$.
=#

using JuMP: is_solved_and_feasible

m = 2
c, model = estimate_c(σ, a, b, m, forward)
@info is_solved_and_feasible(model)

#=
We check that there is an admissible vector $c$ for $m=3$.
=#

m = 3
c, model = estimate_c(σ, a, b, m, forward)
@info is_solved_and_feasible(model)
@info c

# ## Resolution of the convex nonlinear SDP

#=
TODO
=#

# ## References

#=
```@bibliography
Pages = ["nonlinear_sdp.md"]
Canonical = false
```
=#