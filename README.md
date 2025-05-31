# AutoDiffLib - A library to compute gradients fast!

AutoDiffLib is a library for building computational graphs. It allows user to superpass it with function `backward!` to count gradient. 
### Library has features like:
 - Building computational graphs with usage of `Variable` and `Constant` object
  - `@toposort` macro for topological graph sort
  - Traversing graph with `backward!` method to compute gradients
  - Various operations on Scalars, Vectors and Matrixes

## Installation
To install use the block of code below:
```julia
using Pkg
Pkg.add('AutoDiffLib')
```
Now you have set of needed functions installed :).

## Example
To use library you have define your parameters as `Variables` and `Constants`:
```julia
W = Variable(randn(out_features, in_features) * sqrt(2 / in_features), "W")
b = Variable(zeros(out_features, 1), "b")

x = Variable([1.0, 2.0], "x")
```

Then you can use `@toposort` macro and `backward!` function to compute gradients:
```julia
z = @toposort Sigmoid(W * x + b)
backward!(z)
```

Now you have access to gradients of each Variable with field grad:
```julia
println(W.grad)
println(b.grad)
println(x.grad)
```

Use your computed gradients as needed :).

## Licence
Project is under licence. Please do not reuse and repost without consent.