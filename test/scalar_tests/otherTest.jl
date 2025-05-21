@testset "Other Scalar" begin
    x_data = 0.5
    x = Variable(x_data, "x")
    
    z = Sigmoid(x)
    backward(z)

    @test x.grad ≈ 0.235 atol=0.001

    x_data = 0.5
    x = Variable(x_data, "x")
    
    z = ReLU(x)
    backward(z)

    @test x.grad == 1

    x_data = -0.5
    x = Variable(x_data, "x")
    
    z = ReLU(x)
    backward(z)

    @test x.grad == 0
end