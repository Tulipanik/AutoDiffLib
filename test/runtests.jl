using AutoDiffLib
using Test

include("scalar_tests/additionTest.jl")
include("scalar_tests/substractionTest.jl")
include("scalar_tests/multiplicationTest.jl")
include("scalar_tests/divisionTest.jl")
include("scalar_tests/exponentioationTest.jl")
include("scalar_tests/logarithmTest.jl")
include("scalar_tests/trigonometryTest.jl")
include("scalar_tests/maxTest.jl")
include("scalar_tests/otherTest.jl")

include("./matrix_tests/additionTest.jl")
include("matrix_tests/substractionTest.jl")
include("matrix_tests/multiplicationTest.jl")
include("matrix_tests/divisionTest.jl")

@testset "Wrapper Test" begin
    f(x, y) = x^2 + 3*y - 2
    params = [2, 3]

    grad = gradient(f, params)
    @test 4 == grad[1]
    @test 3 == grad[2]

    f(x, y) = x + y + x
    params = [[1.0 2.0; 3.0 4.0], [5.0 6.0; 7.0 8.0]]

    grad = gradient(f, params)
    @test grad[1] == [2.0 2.0; 2.0 2.0]
    @test grad[2] == [1.0 1.0; 1.0 1.0]

end