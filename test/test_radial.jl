using RadialCalderon: build_system, solve_system, forward_map

@testset "constructor" begin
    r = [0.5, 0.25]
    problem = ForwardProblem(r)

    @test problem.n == 3
end

@testset "build and solve system" begin
    r = [0.5, 0.25]
    n = length(r) + 1
    problem = ForwardProblem(r)
    σ = ones(n)

    L = build_system(problem, 1, σ)

    @test size(L) == (2*(n-1), 2*(n-1))

    αβ = solve_system(problem, 1, σ)

    @test length(αβ) == size(L, 1)

    # test coherence with Harrach's computations
    @test isapprox(αβ, [1.0, 0.0, 1.0, 0.0])
   
    # same test with σ different from ones(n) and j different from 1
    σ1 = 1.5
    σ2 = 0.5
    σ = [1.0, σ1, σ2]
    j = 2
    αβ = solve_system(problem, j, σ)

    α1 = 0.25 * ((1/σ1+1)*(σ1+σ2) + (1/σ1-1)*(σ1-σ2)*r[2]^(2*j)/r[1]^(2*j))
    @test αβ[1] ≈ α1
    β1 = 0.25 * ((1/σ1-1)*(σ1+σ2)*r[1]^(2*j) + (1/σ1+1)*(σ1-σ2)*r[2]^(2*j))
    @test αβ[2] ≈ β1
    α2 = 0.5 * (1+σ2/σ1)
    @test αβ[3] ≈ α2
    β2 = 0.5 * (1-σ2/σ1) * r[2]^(2*j)
    @test αβ[4] ≈ β2

    # test that introducing a fictious annulus does not modify results
    r = [0.5, 0.3, 0.25]
    problem = ForwardProblem(r)
    σ = [1.0, 1.5, 1.5, 0.5]
    j = 2
    αβ_bis = solve_system(problem, j, σ)

    @test all(αβ_bis[1:4] .≈ αβ[1:4])
    @test all(αβ_bis[5:6] .≈ αβ[3:4])
end

@testset "forward map" begin
    r = [0.5, 0.25]
    problem = ForwardProblem(r)
    σ1 = 1.5
    σ2 = 0.5
    σ = [1.0, σ1, σ2]

    # test coherence with Harrach's computations
    c(j) = (1/σ1+1)*(σ1+σ2) + (1/σ1-1)*(σ1-σ2)*r[2]^(2*j)/r[1]^(2*j)
    d(j) = (1/σ1-1)*(σ1+σ2)*r[1]^(2*j) + (1/σ1+1)*(σ1-σ2)*r[2]^(2*j)

    @test forward_map(problem, 1, σ) ≈ (c(1)+d(1)) / (c(1)-d(1))
    @test forward_map(problem, 2, σ) ≈ (c(2)+d(2)) / (2*(c(2)-d(2)))
    @test forward_map(problem, 3, σ) ≈ (c(3)+d(3)) / (3*(c(3)-d(3)))

    # test equality with 1/(j*σ0) for a constant conductivity equal to σ0
    σ0 = 0.5 
    for j=1:3
        @test forward_map(problem, j, σ0.*ones(problem.n)) == 1/(j*σ0)
    end
end