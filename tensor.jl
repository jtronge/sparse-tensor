# Load the tensor nonzeros from a FROSTT file
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

# Function based on SPLATT's p_find_layer_boundaries()
function find_layer_boundaries(tensor, tensor_dims, proc_dims, mode, slice_sizes)
    layer_dim = proc_dims[mode]
    taget_nnz = length(tensor) ÷ layer_dim
    # Initialize the layer pointers
    layer_ptrs = [0 for i=1:(layer_dim + 1)]
    layer_ptrs[layer_dim + 1] = tensor_dims[mode]
    # Drop out early for dimensions of 1
    if layer_dim == 1
        return layer_ptrs
    end
    # Loop variables
    curr_proc = 1
    last_nnzcount = 0
    nnzcount = slice_sizes[mode][1]
    for slice=2:tensor_dims[mode]
        if nnzcount >= (last_nnzcount + target_nnz)
            # Pick one of the slices to assign, depending on count
            thisdist = nnzcount - (last_nnzcount + target_nnz)
        end
    end
    # Desired number of nonzeros per logical proces slayer
    return layer_ptrs
end

# Build out the entire process grid.
function create_proc_grid(proc_dims)
    function create_proc_grid_rec!(mode, co, proc_grid)
        if mode > length(proc_dims)
            push!(proc_grid, co)
            return
        end
        for i=1:proc_dims[mode]
            new_co = copy(co)
            push!(new_co, i)
            create_proc_grid_rec!(mode + 1, new_co, proc_grid)
        end
    end

    proc_grid = []
    create_proc_grid_rec!(1, [], proc_grid)
    return proc_grid
end

struct Process
    coords::Array{Int64}
    layer_starts::Array{Int64}
    layer_ends::Array{Int64}
end

function create_procs(proc_dims, layer_ptrs)
    proc_coords = create_proc_grid(proc_dims)
    procs = []
    for co in proc_coords
        layer_starts = []
        layer_ends = []
        for m=1:length(proc_dims)
            push!(layer_starts, layer_ptrs[m][co[m]])
            push!(layer_ends, layer_ptrs[m][co[m] + 1])
        end
        proc = Process(co, layer_starts, layer_ends)
        push!(procs, proc)
    end
    return procs
end

# Return the number of rows to communicate for some rank given a tensor and
# total number of ranks.
function communication(tensor, rank, nranks)
    tensor_dims = get_tensor_dims(tensor)
    proc_dims = compute_processor_dimensions(tensor_dims, nranks)
    println(proc_dims)

    ssizes = count_nonzeros_per_slice(tensor, tensor_dims)
    layer_ptrs = []
    for m=1:length(tensor_dims)
        push!(layer_ptrs, find_layer_boundaries(tensor, tensor_dims, proc_dims,
                                                m, ssizes))
    end

    procs = create_procs(proc_dims, layer_ptrs)
    println(procs)
    # println(ssizes[2])

    # Based on p_rearrange_medium()
    # TODO: Next, assign nonzeros to processors
end
