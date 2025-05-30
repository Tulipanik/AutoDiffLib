@testset "Exponention Scalar" begin
    x_data = 2.0
    y_data = 3.0
    x = Variable(x_data, "x")
    y = Variable(y_data, "y")
    
    z = @toposort x^y
    backward!(z)
    
    @test x.grad == 12.0
    @test y.grad ≈ 5.5451 atol=0.001

    x_data = 2.0
    x = Variable(x_data, "x")
    
    z = @toposort exp(x)
    backward!(z)

    @test x.grad ≈ exp(x_data)
end