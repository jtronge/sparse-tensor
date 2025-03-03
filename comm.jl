include("tensor.jl")

tensor = load_tensor("lbnl-network.tns")
communication(tensor, 0, 24)
