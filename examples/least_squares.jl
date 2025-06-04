# # Reconstruction via root finding and least squares

#=
In this tutorial, we show how to implement a root finding or least squares approach to solve the Calderón problem in the noiseless setting. To be more precise, we wish to solve the equation $\Lambda(\sigma)=\Lambda(\sigma^\dagger)$, where $\sigma^\dagger$ is an unknown conductivity. Since $\Lambda$ is nonlinear, iterative approaches could in principle suffer from the problem of local convergence (their output might heavily depend on their initialization). However, we will see that, in practice, these methods yield good results and seem to converge to a solution regardless of their initialization.
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

# ## Root finding

#=
We use the package [NonlinearSolve](https://github.com/SciML/NonlinearSolve.jl) to solve the equation with the Newton-Raphson method initialized with a random guess $\sigma_{\mathrm{init}}$. We notice that, in practice, the method converges to a solution regardless of the initialization.
=#

using NonlinearSolve

residual(σ, p) = Λ(σ) .- obs_true
max_linf_err = 0.0

for i=1:30
    σ_init = a .+ (b-a) .* rand(n)  # random initialization
    problem = NonlinearProblem(residual, σ_init)
    res = NonlinearSolve.solve(problem, NewtonRaphson(), abstol=1e-12)
    σ_hat = res.u
    global max_linf_err = max(max_linf_err, maximum(abs.(σ_hat .- σ_true)))
end

max_linf_err

# ## Least squares

#=
The same package can be used to solve the equation by minimizing the least squares functional 

$\sigma\mapsto \frac{1}{2}\| \Lambda(\sigma) - \Lambda(\sigma^\dagger) \|_2^2$

with an iterative optimization algorithm (here, we use the Levenberg-Marquardt algorithm).
=#

max_linf_err = 0.0

for i=1:30
    σ_init = a .+ (b-a) .* rand(n)  # random initialization
    problem = NonlinearLeastSquaresProblem(residual, σ_init)
    res = NonlinearSolve.solve(problem, LevenbergMarquardt(), abstol=1e-12)
    σ_hat = res.u
    global max_linf_err = max(max_linf_err, maximum(abs.(σ_hat .- σ_true)))
end

max_linf_err

#=
We can also do the same experiment by using the BFGS algorithm of the [Optim](https://github.com/JuliaNLSolvers/Optim.jl) package via the [Optimization](https://github.com/SciML/Optimization.jl) interface.
=#

using Optimization
using OptimizationOptimJL

obj(σ, p) = 0.5 * sum((Λ(σ) .- obs_true).^2)
optfun = OptimizationFunction(obj, Optimization.AutoForwardDiff())

max_linf_err = 0.0

for i=1:30
    σ_init = a .+ (b-a) .* rand(n)  # random initialization
    problem = OptimizationProblem(optfun, σ_init)
    res = solve(problem, Optim.BFGS(), g_tol=1e-13)
    σ_hat = res.u
    global max_linf_err = max(max_linf_err, maximum(abs.(σ_hat .- σ_true)))
end

max_linf_err