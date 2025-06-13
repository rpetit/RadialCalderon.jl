using RadialCalderon
using Test
using Aqua

@testset "aqua" begin
    Aqua.test_all(RadialCalderon)
end

@testset "forward" begin
    include("test_forward.jl")
end

@testset "nonlinear SDP" begin
    include("test_nonlinear_sdp.jl")
end