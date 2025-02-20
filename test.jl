include("tensor.jl")

# Check against known decomposition values
@assert compute_processor_dimensions([4821207, 1774269, 1805187], 384) == [12, 4, 8]
@assert compute_processor_dimensions([46, 239172, 239172], 384) == [1, 24, 16]
@assert compute_processor_dimensions([8211298, 176962, 8116559], 48) == [6, 1, 8]
