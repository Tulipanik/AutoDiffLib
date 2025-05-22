function gradient(f::Function, params::AbstractArray)
    variables = [Variable(p, "x$i") for (i, p) in enumerate(params)]

    output = f(variables...)
    backward(output)

    return [v.grad for v in variables]
end

function backward(output::Node{T}) where {T}
    if hasfield(typeof(output), :grad)
        output.grad = init_grad(output.value)
    end

    visited = Set{Node{T}}()
    order = topological_sort(output, visited, Vector{Node{T}}())
    reversed_order = reverse(order)

    for node in reversed_order
        if node isa Operation
            grads = node.backward(node.grad)
            for (input, grad) in zip(node.inputs, grads)
                if hasfield(typeof(input), :grad)
                    input.grad += grad
                end
            end
        end
    end
end


init_grad(value::Number) = one(value)
init_grad(value::AbstractArray) = ones(size(value))

function reset_grad!(node::Node{T}, visited::Set{Node{T}} = Set{Node{T}}()) where {T}
    if node in visited
        return
    end
    push!(visited, node)
    node.grad = zero(node.grad)
    if node isa Operation{T}
        for input in node.inputs
            reset_grad!(input, visited)
        end
    end
end
