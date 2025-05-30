@testset "Division Matrix" begin
    x_data = [1.0 2.0; 3.0 4.0]
    y_data = [5.0 6.0; 7.0 8.0]
    x = Variable(x_data, "x")
    y = Variable(y_data, "y")
    
    z = @toposort x / y
    backward!(z)

    @test x.grad ≈ [-1.0 1.0; -1.0 1.0] atol=0.01
    @test y.grad ≈ [4.0 -4.0; 6.0 -6.0] atol=0.01

    x_data = [1.0 2.0; 3.0 4.0]
    y_data = [5.0 6.0; 7.0 8.0]
    x = Variable(x_data, "x")
    y = Variable(y_data, "y")
    
    z = @toposort x ./ y ./ x
    backward!(z)

    @test x.grad == [0.0 0.0; 0.0 0.0]
    @test y.grad ≈ [-0.04 -0.0278; -0.0204 -0.0156] atol=0.001

    x_data = [1.0 2.0]
    y_data = [5.0 6.0; 7.0 8.0]
    x = Variable(x_data, "x")
    y = Variable(y_data, "y")
    
    z = @toposort x / y
    backward!(z)

    @test x.grad ≈ [-1.0 1.0] atol=0.001
    @test y.grad ≈ [1.0 -1.0; 2.0 -2.0] atol=0.001

    x_data = [1.0 2.0]
    y_data = [5.0 6.0]
    x = Variable(x_data, "x")
    y = Variable(y_data, "y")
    
    z = @toposort x ./ y
    backward!(z)

    @test x.grad ≈ [0.2 0.1667] atol=0.001
    @test y.grad ≈ [-0.0400 -0.0556] atol=0.001
    
end
