# # Reconstruction via nonlinear least squares

#=
In this tutorial, we show how to implement the least squares approach to solve the Calderón problem in the noiseless setting. To be more precise, we wish to minimize the functional 

$f:\sigma\mapsto \frac{1}{2}\|\Lambda(\sigma)-\Lambda(\sigma^\dagger)\|_2^2$

where $\sigma^\dagger$ is an unknown conductivity. Since $\Lambda$ is nonlinear, $f$ is non-convex and iterative minimization algorithms could in principle suffer from the problem of local convergence (their output might heavily depend on their initialization). However, we will see that, in practice, the main issue is rather the ill-posedness of the inverse problem, and that robust iterative algorithms almost always converge to a global minmizer regardless of their initialization.
=#

# ## Setting

using RadialCalderon

n = 3
r = [0.5, 0.25]
forward = ForwardProblem(r)

m = n
Λ(σ) = [forward_map(forward, j, σ) for j in 1:m]  # forward map

a = 0.5
b = 1.5

σ_true = [0.8, 1.2, 1.0]  # unknown conductivity
obs_true = Λ(σ_true);  # observations

# ## Using [NonlinearSolve.jl](https://github.com/SciML/NonlinearSolve.jl)

#=
We use the package [NonlinearSolve.jl](https://github.com/SciML/NonlinearSolve.jl) to solve the equation $\Lambda(\sigma)=\Lambda(\sigma^\dagger)$ using a robust Newton-type algorithm initialized with a guess $\sigma_{\mathrm{init}}$. We notice that, in practice, the method converges to a solution regardless of its initialization.
=#

using Base.Iterators: product

k = 5
prod_it = product([range(a, b, k) for i=1:3]...)
σ_init_tab = hcat(collect.(collect(prod_it))...)
nσ_init = size(σ_init_tab, 2)

@info "Number of initializations: $nσ_init"

#=
The set of initial guesses is a regular discretization of $[a,b]^n$ using $k^n$ points (here, we chose $k=5$).
=#

using NonlinearSolve

residual(σ, p) = Λ(σ) .- obs_true
max_linf_err = 0.0
max_linf_err_obs = 0.0

for i=1:nσ_init
    σ_init = σ_init_tab[:, i]
    problem = NonlinearProblem(residual, σ_init)
    res = NonlinearSolve.solve(problem, RobustMultiNewton(), abstol=1e-15)
    σ_hat = res.u
    global max_linf_err = max(max_linf_err, maximum(abs.(σ_hat .- σ_true)))
    global max_linf_err_obs = max(max_linf_err_obs, maximum(abs.(Λ(σ_hat) .- obs_true)))
end

@info "Maximum l-infinity error (conductivity): $max_linf_err"
@info "Maximum l-infinity error (observations): $max_linf_err_obs"

# ## Using [Optimization.jl](https://github.com/SciML/Optimization.jl)

#=
We can also use the [Optimization.jl](https://github.com/SciML/Optimization.jl) package to call a large list of optimization algorithms to minimize $f$. Here, we use an interior point Newton algorithm from the [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl) package. We enforce the constraint $\sigma_i>10^{-5}$ to avoid domain errors.
=#

using Optimization
using OptimizationOptimJL
using DifferentiationInterface
import ForwardDiff

obj(σ, p) = 0.5 * sum((Λ(σ) .- obs_true).^2)
optfun = OptimizationFunction(obj, SecondOrder(AutoForwardDiff(), AutoForwardDiff()))

max_linf_err = 0.0
max_linf_err_obs = 0.0

for i=1:nσ_init
    σ_init = σ_init_tab[:, i]
    problem = OptimizationProblem(optfun, σ_init, lb=1e-5.*ones(n), ub=Inf.*ones(n))
    res = solve(problem, IPNewton(), g_tol=1e-17)
    σ_hat = res.u
    global max_linf_err = max(max_linf_err, maximum(abs.(σ_hat .- σ_true)))
    global max_linf_err_obs = max(max_linf_err_obs, maximum(abs.(Λ(σ_hat) .- obs_true)))
end

@info "Maximum l-infinity error (conductivity): $max_linf_err"
@info "Maximum l-infinity error (observations): $max_linf_err_obs"