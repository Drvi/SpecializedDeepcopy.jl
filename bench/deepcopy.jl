# Compare copies of model parameter graphs: julia --project bench/deepcopy.jl.
# Report the lowest batch-average time per copy over one second, plus bytes allocated per copy.
using SpecializedDeepcopy

mutable struct Owner
    id::Float64
end

mutable struct Node
    label::String
    next::Union{Nothing, Node}
end

struct Table
    t::Vector{Float64}
    u::Vector{Float64}
    last_index::Base.RefValue{Int}
    derivative::Union{Nothing, Vector{Float64}}
end

struct Medium
    name::String
    data::@NamedTuple{critical::@NamedTuple{p::Float64, T::Float64}, coefficients::Matrix{Float64}, phases::Vector{Symbol}}
    tables::Dict{Symbol, Table}
    unit::Union{Nothing, Symbol}
    limits::Tuple{Float64, Union{Missing, Float64}}
end

function medium(i)
    table = Table(collect(0.0:0.1:1.0), rand(11), Ref(1), iseven(i) ? nothing : rand(11))
    data = (critical = (p = 5.78e6, T = 351.3), coefficients = rand(4, 4), phases = [:liquid, :vapor])
    return Medium("R$i", data, Dict(:density => table, :enthalpy => table), :SI, (0.0, missing))
end

function linked_list(n)
    node = Node("end", nothing)
    for _ in 1:(n - 1)
        node = Node("node", node)
    end
    return node
end

const CASES = [
    "numeric buffers" => (tunable = rand(10_000), discrete = (rand(100), rand(Int, 100)), constant = (rand(10),)),
    "small value" => (rand(10), Owner(1.0), "name"),
    "10 000 mutable structs" => [Owner(i) for i in 1:10_000],
    "100 media" => [medium(i) for i in 1:100],
    "Dict of 1 000 vectors" => Dict("key$i" => rand(10) for i in 1:1000),
    "Symbol-keyed Dict" => Dict(Symbol("key$i") => rand(10) for i in 1:1000),
    "Dict of 1 000 mutable keys" => Dict(Owner(i) => rand(10) for i in 1:1000),
    "1 000 immutable tables" => [Table(rand(11), rand(11), Ref(1), iseven(i) ? nothing : rand(11)) for i in 1:1000],
    "10 000 structs as Any" => Any[Owner(i) for i in 1:10_000],
    "list of 1 000 nodes" => linked_list(1000),
    "10 000 Union elements" => Union{Missing, Float64}[isodd(i) ? missing : Float64(i) for i in 1:10_000],
    "1 000 strings" => ["string$i" for i in 1:1000],
]

# Warm up, estimate a ~100 µs batch to reduce timer granularity effects, then sample for one second.
function fastest(f, x)
    f(x)
    batch = max(1, ceil(Int, 100e-6 / (@elapsed f(x))))
    best = Inf
    start = time()
    while time() - start < 1.0
        elapsed = @elapsed for _ in 1:batch
            f(x)
        end
        best = min(best, elapsed / batch)
    end
    return best
end

allocated(f, x) = (f(x); @allocated f(x))

column(x, width) = lpad(x, width)
microseconds(t) = string(round(t * 1e6; digits = 2), " µs")

# Sys.CPU_NAME is LLVM's target label, such as apple-m1 even on newer Apple chips.
# cpu_info provides the processor model reported by the OS.
println("Julia ", VERSION, " on ", Sys.cpu_info()[1].model)
# The first column fits the longest case name.
const NAME_WIDTH = maximum(length(name) for (name, _) in CASES) + 2
println(rpad("value", NAME_WIDTH), column("deepcopy", 14), column("specialized", 14), column("speedup", 9), column("deepcopy", 12), column("specialized", 12))
for (name, x) in CASES
    base, special = fastest(deepcopy, x), fastest(specialized_deepcopy, x)
    println(rpad(name, NAME_WIDTH), column(microseconds(base), 14), column(microseconds(special), 14), column(string(round(base / special; digits = 2), "x"), 9),
        column(string(allocated(deepcopy, x), " B"), 12), column(string(allocated(specialized_deepcopy, x), " B"), 12))
end
