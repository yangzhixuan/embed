#= linear softmax classifier (with stochastic gradient descent)
   LinearClassifier(k, n)
   train(c, X, y)
   train_one(c, x, y)
   accuracy(c, X, y)
   predict(c, x)
=#

using NumericExtensions
using IProfile

type LinearClassifier
    k :: Int64 # number of outputs
    n :: Int64 # number of inputs
    weights :: Array{Float64, 2} # k * n weight matrix

    outputs :: Vector{Float64}
end

function LinearClassifier(k, n)
    weights = rand(n, k) * 2 - 1; # range is [-1, 1]
    LinearClassifier(k, n, weights, zeros(k))
end

function predict(c :: LinearClassifier, x :: Array{Float64})
    # the softmax() function from the NumericExtension package is more numeric stable
    return vec(softmax(x * c.weights))
end

function predict!(c :: LinearClassifier, x :: Array{Float64})
    c.outputs = vec(softmax(x * c.weights))
end

@iprofile begin
function train_one(c :: LinearClassifier, x :: Array{Float64}, y :: Int64; α :: Float64 = 0.025, input_gradient :: Union(Nothing, Array{Float64}) = nothing)
    if !in(y, 1 : c.k)
        msg = @sprintf "A sample is discarded because the label y = %d is not in range of 1 to %d" y c.k
        warn(msg)
        return
    end

    predict!(c, x)
    c.outputs[y] -= 1

    if input_gradient != nothing
        # input_gradient = ( c.weights * outputs' )'
        BLAS.gemv!('N', α, c.weights, c.outputs, 1.0, input_gradient)
    end

    # c.weights -= α * x' * outputs;
    BLAS.ger!(-α, vec(x), c.outputs, c.weights)
end
end

# calculate the overall log likelihood. Mainly used for debugging
function log_likelihood(c, X, y)
    n = size(X, 1)
    l = 0
    for i in 1 : n
        l += log(predict(c, X[i, :])[y[i]]);
    end
    return l
end

# calculate the accuracy on the testing dataset
function accuracy(c :: LinearClassifier, X :: Array{Float64}, y :: Array{Int64})
    n = size(X, 1)
    succ = 0
    for i in 1 : n
        output = predict(c, X[i, :])
        if maximum(output) == output[y[i]]
            succ += 1
        end
    end
    return succ / n
end


# train on the whole dataset by stochastic gradient descent.
function train_parallel(c, X, y; threshold = 1e-4, max_iter = 100)
    function work(tuple)
        (c, X, y) = tuple
        n = size(X, 1)
        for j in 1:max_iter
            @printf "%d-th iteration(%d)\n" j n
            for i in 1:n
                train_one(c, X[i, :], y[i])
            end
        end
        c.weights
    end

    n = size(X, 1)
    l = log_likelihood(c, X, y)
    @printf "overall log-likelihood: %f\n" l

    number_workers = nworkers()
    parts = partition(shuffle(collect(1:n)), number_workers)
    c.weights = @parallel (.+) for ind in parts
        work((c, X[ind, :], y[ind]))
    end

    c.weights /= number_workers

    new_l = log_likelihood(c, X, y)
    @printf "overall log-likelihood: %f\n" new_l
end
