function ReLU(x::Number)
    if x >= 0
        return x 
    end
    return 0
end

function Sigmoid(x::Number)
    return 1/(1+exp(-x))
end