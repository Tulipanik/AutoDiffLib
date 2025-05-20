@testset "Division Matrix" begin
    x_data = [1.0 2.0; 3.0 4.0]
    y_data = [5.0 6.0; 7.0 8.0]
    x = Variable(x_data, "x")
    y = Variable(y_data, "y")
    
    z = x / y
    backward(z)

    @test x.grad == [0.2 0.1666; 0.1429 0.125] atol=0.001
    @test y.grad == -[0.04 0.0556; 0.0612 0.0625] atol0.001
end
