# # Convergence of least squares solvers

#=
In this experiment, we wish to investigate when least squares solvers fail to converge. We 
found that the best choice of solver is Powell's dog leg method, which is a trust-region 
method combining the steepest and the Gauss-Newton descent directions. We found that this solver 
always converges for $n\leq 9$, but sometimes runs for a very large number of iterations. For 
$n=10$, it fails to converge on very few instances (less than 1 in 100), and we conjecture 
that allowing for more iterations would lead to convergence. In any case, we never found an 
instance where the solver converges to a point which is not an approximate global 
minimizer of the least squares objective.
=#

# ## Setting

using RadialCalderon
using NonlinearSolve
using Random
using Plots
using LaTeXStrings

using DifferentiationInterface
import ForwardDiff

backend = AutoForwardDiff()  # autodiff backen

Random.seed!(1234)  # fix the random seed for reproducibility

a = 0.5  # conductivity lower bound
b = 1.5  # conductivity upper bound

nσ_true = 100  # number of true conductivities
offset = 1e-5  # parameter for the smoothed absolute value
solver = TrustRegion();  # trust-region dogleg solver

# ## Check convergence for $n\leq 9$

σ_true_tab_9 = zeros(9, nσ_true)  # save results for $n=9$
σ_init_tab_9 = zeros(9, nσ_true)
iter_tab_9 = zeros(Int, nσ_true)

for n=1:9
    @info "n=$n"
    m = n

    σ_true_tab = a .+ (b-a) .* rand(n, nσ_true)
    σ_init_tab = a .+ (b-a) .* rand(n, nσ_true)
    
    σ_hat_tab = zeros(n, nσ_true)
    iter_tab = zeros(Int, nσ_true)
    err_tab = zeros(nσ_true)
    err_tab_measurements = zeros(nσ_true)

    r = reverse([i/n for i=1:n-1])
    forward = ForwardProblem(r)
    Λ(σ) = [forward_map(forward, j, σ) for j=1:m]

    for iσ_true=1:nσ_true
        σ_true = σ_true_tab[:, iσ_true]
        σ_init = σ_init_tab[:, iσ_true]
        obs_true = Λ(σ_true)

        f(σ, p) = Λ(sqrt.(σ.^2 .+ offset)) .- obs_true  # re-parameterization to avoid non-positive conductivities

        problem = NonlinearProblem(f, σ_init)
        res = NonlinearSolve.solve(problem, solver, abstol=1e-15, maxiters=200000)
        σ_hat = sqrt.((res.u).^2 .+ offset)

        σ_hat_tab[:, iσ_true] .= σ_hat
        iter_tab[iσ_true] = res.stats.nsteps

        err = abs.(σ_hat .- σ_true) ./ abs.(σ_true)
        err_tab[iσ_true] = maximum(err)
        err_tab_measurements[iσ_true] = maximum(abs.(Λ(σ_hat) .- obs_true))
    end

    @info "maximum relative error (conductivity): $(maximum(err_tab))"
    @info "maximum error (measurements): $(maximum(err_tab_measurements))"

    if n == 9
        σ_true_tab_9 .= σ_true_tab
        σ_init_tab_9 .= σ_init_tab
        iter_tab_9 .= iter_tab
    end
end

# ## Investigate when the number of iterations is large

n = 9
m = n

r = reverse([i/n for i=1:n-1])
forward = ForwardProblem(r)
Λ(σ) = [forward_map(forward, j, σ) for j=1:m]

iσ_true = argmax(iter_tab_9)
σ_true = σ_true_tab_9[:, iσ_true]
σ_init = σ_init_tab_9[:, iσ_true]
obs_true = Λ(σ_true)

f(σ, p) = Λ(sqrt.(σ.^2 .+ offset)) .- obs_true

problem = NonlinearProblem(f, σ_init)
res = NonlinearSolve.solve(
    problem, 
    solver, 
    abstol=1e-15, 
    maxiters=200000,
    store_trace=Val(true),
    trace_level=TraceAll()
)

σ_hat = sqrt.((res.u).^2 .+ offset)

obj(σ) = 0.5 * sum((Λ(σ) .- obs_true).^2)
grad_obj(σ) = gradient(obj, backend, σ)

length_trace = length(res.trace.history)
σ_tab = [sqrt.((res.trace.history[i].u).^2 .+ offset) for i=1:length_trace]
residual_tab = [maximum(abs.(Λ(σ_tab[i]) .- obs_true)) for i=1:length_trace]
dist_tab = [maximum(abs.(σ_tab[i] .- σ_true)) for i=1:length_trace]
grad_norm_tab = [maximum(abs.(grad_obj(σ_tab[i]))) for i=1:length_trace]

plot(
    residual_tab, 
    xscale=:log10, 
    yscale=:log10, 
    xlabel=L"\mathrm{iteration}", 
    label="residual norm",
    c=:blue, 
    yticks = 10 .^ (-16.0:2.0:10.0), 
    xticks = 10 .^ (0:1:6), 
    legend=:bottomleft,
    lw=2,
    xtickfontsize=15,
    ytickfontsize=15,
    legendfontsize=12,
    labelfontsize=16,
    formatter=:latex
)

plot!(
    dist_tab,
    label="distance to solution",
    c=:red, 
    lw=2
)

plot!(
    grad_norm_tab,
    label="gradient norm",
    c=:green, 
    lw=2
)

#=
We see that, for the instance with the largest number of iterations, the iterates go very 
far away from the true conductivity while keeping a low value for the residual. After a very 
large number of iterations, the iterates finally converge to the true conductivity. This is 
likely due to the ill-posedness of the problem, which allows having very different 
conductivities while having similar measurements.
=#

# ## Check convergence for $n=10$

n = 10
m = n

σ_true_tab = a .+ (b-a) .* rand(n, nσ_true)
σ_init_tab = a .+ (b-a) .* rand(n, nσ_true)

σ_hat_tab = zeros(n, nσ_true)
iter_tab = zeros(Int, nσ_true)
err_tab = zeros(nσ_true)
err_tab_measurements = zeros(nσ_true)

r = reverse([i/n for i=1:n-1])
forward = ForwardProblem(r)
Λ(σ) = [forward_map(forward, j, σ) for j=1:m]

maxiters = 500000

for iσ_true=1:nσ_true
    local σ_true = σ_true_tab[:, iσ_true]
    local σ_init = σ_init_tab[:, iσ_true]
    local obs_true = Λ(σ_true)

    f(σ, p) = Λ(sqrt.(σ.^2 .+ offset)) .- obs_true

    local problem = NonlinearProblem(f, σ_init)
    local res = NonlinearSolve.solve(problem, solver, abstol=1e-15, maxiters=maxiters)
    local σ_hat = sqrt.((res.u).^2 .+ offset)

    σ_hat_tab[:, iσ_true] .= σ_hat
    iter_tab[iσ_true] = res.stats.nsteps

    err = abs.(σ_hat .- σ_true) ./ abs.(σ_true)
    err_tab[iσ_true] = maximum(err)
    err_tab_measurements[iσ_true] = maximum(abs.(Λ(σ_hat) .- obs_true))
end

@info "number of convergence failures: $(sum(err_tab_measurements .> 1e-15)) out of $(nσ_true)"

for iσ_true=1:nσ_true
    if err_tab_measurements[iσ_true] > 1e-15
        @info "instance $iσ_true: $(iter_tab[iσ_true]) iterations (max $maxiters)"
    end
end

#=
The only instances for which the solver fails to converge have reached the maximum number of 
iterations.
=#