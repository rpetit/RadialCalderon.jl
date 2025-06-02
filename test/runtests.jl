using RadialCalderon
using Test
using Aqua

@testset "Aqual.jl" begin
    Aqua.test_all(RadialCalderon)
end

@testset "Forward" begin
    include("test_forward.jl")
end

@testset "Nonlinear SDP" begin
    include("test_nonlinear_sdp.jl")
end