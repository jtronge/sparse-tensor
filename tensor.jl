function load_tensor(fname)
    tensor = []
    for line in eachline(fname)
        line = strip(line)
        if startswith(line, '#')
            continue
        end
        nonzero = collect(eachsplit(line))
        co = map(s -> parse(Int, s), nonzero[1:end-1])
        value = parse(Float64, nonzero[end])
        push!(tensor, tuple(co, value))
        if length(tensor) % 1000000 == 0
            println("loaded ", length(tensor), " nonzeros")
        end
    end
    println("loaded ", length(tensor), " nonzeros in total")
    return tensor
end

# Compute the prime factors of a number n.
function get_primes(n)
    primes = []
    while n != 1
        next_prime = 1
        for i=2:n
            if n % i == 0
                next_prime = i
                break
            end
        end
        push!(primes, next_prime)
        n ÷= next_prime
    end
    return primes
end

function compute_processor_dimensions(tensor_dims, nranks)
    # First compute the primes for nnodes
    primes = get_primes(nranks)
    total_nnz = sum(tensor_dims)
    target_nnz = total_nnz ÷ nranks
    proc_dims = [1 for i=1:length(tensor_dims)]
    for i=length(primes):-1:1
        # Compute the mode (tensor dimension) contributing the most nonzeros
        furthest = argmax([tdim ÷ pdim for (tdim, pdim) in zip(tensor_dims, proc_dims)])
        proc_dims[furthest] *= primes[i]
    end
    return proc_dims
end

function get_tensor_dims(tensor)
    dims = copy(tensor[1][1])
    for (co, _) in tensor
        for i=1:length(co)
            dims[i] = max(co[i], dims[i])
        end
    end
    return dims
end

function count_nonzeros_per_slice(tensor, tensor_dims)
    ssizes = [[0 for i=1:tensor_dims[m]] for m=1:length(tensor_dims)]
    for m=1:length(tensor_dims)
        for (co, _) in tensor
            ssizes[m][co[m]] += 1
        end
    end
    return ssizes
end

# Return the number of rows to communicate for some rank given a tensor and
# total number of ranks.
function communication(tensor, rank, nranks)
    tensor_dims = get_tensor_dims(tensor)
    proc_dims = compute_processor_dimensions(tensor_dims, nranks)
    println(proc_dims)

    ssizes = count_nonzeros_per_slice(tensor, tensor_dims)
    println(ssizes[2])

    # Based on p_rearrange_medium()
    # TODO: Next, assign nonzeros to processors
end
