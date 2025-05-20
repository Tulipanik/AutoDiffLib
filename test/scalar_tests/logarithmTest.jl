@testset "Logarithm Scalar" begin
    x_data = 2.0
    x = Variable(x_data, "x")
    
    z = log(x)
    backward(z)

    @test x.grad == 0.5
end