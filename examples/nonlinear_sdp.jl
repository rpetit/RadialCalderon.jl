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
For a given $m$, one can try to find a universal vector $c$ by solving a feasibility problem constructed via [`build_c_estimation_problem`](@ref). Below, we check that, when $n=2$, this is possible for $m=3$ but not for $m=2$. First, we define a set of conductivities which will be used to estimate $c$.
=#

using Base.Iterators: product

k = 2

prod_it = product([range(a, b, k) for i=1:forward.n]...)
σ = hcat(collect.(collect(prod_it))...)
σ = σ[:, 1:end-1];  # remove b*ones(n)

#=
We build the $c$ estimation problem for $m=2$ and check that it is not feasible. The problem is a linear program, so that any suitable solver other than Ipopt can be used (in our experiments, MOSEK performed best).
=#

using JuMP
import Ipopt

m = 2
model = build_c_estimation_problem(σ, a, b, m, forward)
optimizer = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0, "sb" => "yes")
set_optimizer(model, optimizer)
optimize!(model)

#=
We check that there is no admissible vector $c$ for $m=2$.
=#

is_solved_and_feasible(model)

#=
We check that there is an admissible vector $c$ for $m=3$.
=#

m = 3
model = build_c_estimation_problem(σ, a, b, m, forward)
set_optimizer(model, optimizer)
optimize!(model)
is_solved_and_feasible(model)

#=
The estimated vector $c$ can be accessed as follows.
=#

c = value.(model[:c])

#=
We stress that, when $n$ is larger, the tolerance of the solver might have to be adjusted to ensure that the problem is solved with good precision.
=#

# ## Resolution of the convex nonlinear SDP

#=
Once the weight vector $c$ is estimated, one can solve the convex nonlinear SDP defined above. To do so, we build the problem using [`build_nonlinear_sdp`](@ref) and use the [Optimization](https://github.com/SciML/Optimization.jl) package with the Ipopt solver.
=#

using Optimization, OptimizationMOI

σ_true = [0.8, 1.2]  # unknown conductivity
obs = [forward_map(forward, j, σ_true) for j=1:m]  # observations
problem = ConvexCalderonProblem(c, a, b, obs, forward)

σ_init = (0.9*b) .* ones(forward.n)  # intial guess
prob = build_nonlinear_sdp(problem, σ_init)  # container for the nonlinear SDP

optimizer = OptimizationMOI.MOI.OptimizerWithAttributes(
    Ipopt.Optimizer,
    "print_level" => 0,
    "sb" => "yes"
)

sol = solve(prob, optimizer)
σ_hat = sol.u

isapprox(σ_hat, σ_true, rtol=1e-6)