using Base.Iterators: product
using JuMP
import Ipopt

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

@testset "estimate c" begin
    r = [0.5]
    forward = ForwardProblem(r)

    a = 0.5
    b = 1.5
    m = 2
    
    k = 2

    prod_it = product([range(a, b, k) for i=1:forward.n]...)
    σ = hcat(collect.(collect(prod_it))...)
    σ = σ[:, 1:end-1]  # remove b*ones(n)

    model = build_c_estimation_problem(σ, a, b, m, forward; max_last_coord=true)
    optimizer = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0, 
                                          "sb" => "yes")
    set_optimizer(model, optimizer)
    optimize!(model)

    # the problem is not feasible for n=2 and m=2
    @test !is_solved_and_feasible(model)

    m = 3
    model = build_c_estimation_problem(σ, a, b, m, forward; max_last_coord=true)
    set_optimizer(model, optimizer)
    optimize!(model)

    # the probme is feasible for n=2 and m=3
    @test is_solved_and_feasible(model)

    c = value.(model[:c])
    @test c[1] ≈ 1.0
    @test c[2] > 0.0 && c[2] < 1.0
end