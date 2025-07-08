# # Comparison of least squares and nonlinear SDP

# ## Setting

using RadialCalderon

using Optimization, OptimizationMOI
using JuMP
import Ipopt

using DifferentiationInterface
import ForwardDiff

using NonlinearSolve

using Plots
using LaTeXStrings

using Random
using Base.Iterators: product

Random.seed!(1234);  # fix the random seed for reproducibility

n = 3
a = 0.5
b = 1.5
m = 7

r = reverse([i/n for i=1:n-1])

forward = ForwardProblem(r)

Λ(σ) = [forward_map(forward, j, σ) for j=1:m]
dΛ(σ) = jacobian(Λ, AutoForwardDiff(), σ);

# ## Estimation of the weight vector $c$

k = 5

prod_it = product([range(a, b, k) for i=1:forward.n]...)
σ = hcat(collect.(collect(prod_it))...)
σ = σ[:, 1:end-1];  # remove b*ones(n)

model = build_c_estimation_problem(σ, a, b, m, forward, max_last_coord=true)
optimizer = optimizer_with_attributes(
    Ipopt.Optimizer, 
    "print_level" => 0, 
    "sb" => "yes",
    "tol" => 1e-13,
    "constr_viol_tol" => 1e-15
)
set_optimizer(model, optimizer)
optimize!(model)
c = value.(model[:c])

@info "c estimation problem solved: $(is_solved_and_feasible(model))"
@info "vector c: $c"

# ## Reconstruction via least squares and nonlinear SDP

nσ_test = 100
σ_test = a .+ (b-a) .* rand(n, nσ_test)
σ_init_sdp = (0.9*b) .* ones(n)

σ_hat_tab_sdp = zeros(n, nσ_test)
σ_hat_tab_ls = zeros(n, nσ_test)

optimizer = OptimizationMOI.MOI.OptimizerWithAttributes(
    Ipopt.Optimizer,
    "print_level" => 0,
    "sb" => "yes",
    "tol" => 1e-15,
    "constr_viol_tol" => 1e-15
)

for iσ_test=1:nσ_test
    σ_true = σ_test[:, iσ_test]
    obs_true = Λ(σ_true)

    convex_calderon = ConvexCalderonProblem(c, a, b, obs_true, forward)
    sdp = build_nonlinear_sdp(convex_calderon, σ_init_sdp)
    sol_sdp = solve(sdp, optimizer)
    σ_hat_sdp = sol_sdp.u
    σ_hat_tab_sdp[:, iσ_test] .= σ_hat_sdp

    f(σ, p) = Λ(σ) .- obs_true
    σ_init_ls = a .+ (b-a) .* rand(n)
    problem = NonlinearProblem(f, σ_init_ls)
    sol_ls = NonlinearSolve.solve(problem, RobustMultiNewton(), abstol=1e-15)
    σ_hat_ls = sol_ls.u
    σ_hat_tab_ls[:, iσ_test] .= σ_hat_ls
end

#=
We can now plot the histogram of the estimation errors for both methods.
=#

linf_err_tab_sdp = vcat(maximum(abs.(σ_hat_tab_sdp .- σ_test), dims=1)...)
linf_err_tab_ls = vcat(maximum(abs.(σ_hat_tab_ls .- σ_test), dims=1)...)

bin = 10.0 .^ (range(-17, -7, 20))
histogram(
    [linf_err_tab_ls, linf_err_tab_sdp],
    bin=bin, 
    xscale=:log10,
    xlim=extrema(bin), 
    ylim=(0, 50),
    xticks=10.0 .^ [-15, -13, -11, -9],
    fillalpha=0.3, 
    fillcolor=[:blue :green], 
    label=["Newton" "SDP"],
    xlabel="Estimation error",
    formatter=:latex, 
    xtickfontsize=20, 
    ytickfontsize=20, 
    legendfontsize=18, 
    labelfontsize=21,
    bottom_margin=5Plots.mm
)