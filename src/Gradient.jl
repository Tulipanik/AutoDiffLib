# function gradient(f::Function, params::AbstractArray)
#     variables = [Variable(p, "x$i") for (i, p) in enumerate(params)]

#     output = @toposort f(variables...)
#     backward!(output)

#     return [v.grad for v in variables]
# end

function backward!(order::Vector{Node})
    last_node = order[end]
    if last_node.grad isa Number
        last_node.grad = one(last_node.value)
    else
        last_node.grad .= one(eltype(last_node.value))
    end

    # @show size(order)

    for node in reverse(order)
        if node isa Operation || node isa SpreadedOperator
            # @show node.op_name
            grads = node.backward(node.grad)
            # @show size(grads)
            # @show size(node.inputs)
            for (input, grad) in zip(node.inputs, grads)
                if input isa Constant
                    continue
                elseif isa(input.grad, Number) && isa(grad, Number)
                    input.grad += grad
                else
                    if size(input.grad) != size(grad)
                        # @show node.op_name
                        # @show size(input.grad)
                        # @show size(grad)
                        grad = reshape(grad, size(input.grad))
                    end
                    input.grad .+= grad
                end
            # @show size(input.grad)
            # @show size(grad)
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
    if node isa Operation{T}
        for input in node.inputs
            reset_grad!(input, visited)
        end
    end
end