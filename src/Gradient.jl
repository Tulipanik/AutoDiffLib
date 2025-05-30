function backward!(order::Vector{Node})
    last_node = order[end]
    if last_node.grad isa Number
        last_node.grad = one(last_node.value)
    else
        last_node.grad .= one(eltype(last_node.value))
    end

    for node in reverse(order)
        if node isa Operation || node isa SpreadedOperator
            grads = node.backward(node.grad)
            for (input, grad) in zip(node.inputs, grads)
                if input isa Constant
                    continue
                elseif isa(input.grad, Number) && isa(grad, Number)
                    input.grad += grad
                else
                    if size(input.grad) != size(grad)
                        grad = reshape(grad, size(input.grad))
                    end
                    input.grad .+= grad
                end
            end
        end
    end
end

function forward!(order::Vector{Node})
    for node in order
        if node isa Operation
            input_vals = (input.value for input in node.inputs)
            node.value = node.foward(input_vals...)
        end
    end
end

init_grad(value::Number) = one(value)
init_grad(value::AbstractArray) = ones(size(value))

function reset_grad!(node::Node{T}, visited::Set{Node{T}}=Set{Node{T}}()) where {T}
    if node in visited
        return
    end
    push!(visited, node)
    node.grad = zero(node.grad)
    if node isa Operation{T} || node isa SpreadedOperator{T}
        for input in node.inputs
            reset_grad!(input, visited)
        end
    end
end