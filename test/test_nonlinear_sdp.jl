using Base.Iterators: product
using JuMP
using Optimization, OptimizationMOI
using Ipopt

@testset "constructor" begin
    r = [0.5, 0.25]
    forward = ForwardProblem(r)

    a = 0.5
    b = 1.5
    m = 2

    σ = a .+ (b-a) .* rand(forward.n)
    obs = [forward_map(forward, j, σ) for j=1:m]

    problem = ConvexCalderonProblem(σ, a, b, obs, forward)
end

@testset "build and solve nonlinear SDP" begin
    r = [0.5]
    forward = ForwardProblem(r)

    a = 0.5
    b = 1.5
    m = 3

    c = [1.0, 0.05]
    σ_true = [0.8, 1.2]
    obs = [forward_map(forward, j, σ_true) for j=1:m]
    problem = ConvexCalderonProblem(c, a, b, obs, forward)

    σ_init = (0.9*b) .* ones(forward.n)
    prob = build_nonlinear_sdp(problem, σ_init)

    optimizer = OptimizationMOI.MOI.OptimizerWithAttributes(
        Ipopt.Optimizer,
        "print_level" => 0,
        "sb" => "yes"
    )

    sol = solve(prob, optimizer)
    σ_hat = sol.u

    @test isapprox(σ_hat, σ_true, rtol=1e-6)
end