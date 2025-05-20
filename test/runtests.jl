using AutoDiffLib
using Test

@testset "AutoDiffLib.jl" begin
    x_data = 2.0
    y_data = 3.0
    x = Variable(x_data, "x")
    y = Variable(y_data, "y")
    
    z = x + y + x

    println(z.grad)
    println(y.grad)
    println(x.grad)
    
    backward(z)
    
    println("dz/dx = ", x.grad)
    println("dz/dy = ", y.grad)
    @test x.grad == 2.0
    @test y.grad == 1.0

    # x_data = [1.0 2.0; 3.0 4.0]
    # y_data = [5.0 6.0; 7.0 8.0]


end
