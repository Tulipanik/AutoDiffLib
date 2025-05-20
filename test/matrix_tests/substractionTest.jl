@testset "Substraction Matrix" begin
    x_data = [1.0 2.0; 3.0 4.0]
    y_data = [5.0 6.0; 7.0 8.0]
    x = Variable(x_data, "x")
    y = Variable(y_data, "y")
    
    z = x - y - x
    backward(z)

    @test x.grad == [0.0 0.0; 0.0 0.0]
    @test y.grad == -[1.0 1.0; 1.0 1.0]
end
