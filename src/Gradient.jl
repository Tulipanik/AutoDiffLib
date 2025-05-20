function backward(output::Node{T}) where {T}
    output.grad = init_grad(output.value)
    visited = Set{Node{T}}()
    order = topological_sort(output, visited, Vector{Node{T}}())
    reversed_order = reverse(order)

    for node in reversed_order
        if node isa Operation{T}
            grads = node.backward(node.grad)
            for (input, grad) in zip(node.inputs, grads)
                input.grad += grad
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
