# A type that holds a WeakKeyDict. The table holds its keys as WeakRefs, which specialized_deepcopy
# rejects: a field copy of a WeakRef bypasses the garbage collector's registration. A method that
# starts an empty table gives the copy its own cache. Wrappers that share a table share its copy.
#
# Run this file alone from the package directory: julia --project test/examples/result_cache.jl
# The test suite includes this file and calls test_result_cache().
using SpecializedDeepcopy
using Test

mutable struct Model
    id::Int
end

# Without a method.
struct ResultCacheWithoutMethod
    results::WeakKeyDict{Model, Vector{Float64}}
end

# With a method that discards the cached results. `copies` keeps one empty table for each original.
struct ResultCache
    results::WeakKeyDict{Model, Vector{Float64}}
end
function SpecializedDeepcopy.specialized_deepcopy(x::ResultCache, copies)
    results = get!(WeakKeyDict{Model, Vector{Float64}}, copies, x.results)::WeakKeyDict{Model, Vector{Float64}}
    return ResultCache(results)
end

function test_result_cache()
    @testset "ResultCache" begin
        model = Model(1)
        @test_throws PointerCopyError(WeakRef, nothing) specialized_deepcopy(ResultCacheWithoutMethod(WeakKeyDict(model => [1.0])))

        cache = ResultCache(WeakKeyDict(model => [1.0]))
        copied = specialized_deepcopy(cache)
        other = Model(2)
        copied.results[other] = [2.0]
        @test length(copied.results) == 1 && length(cache.results) == 1 && !haskey(cache.results, other)

        # A repeated wrapper and another wrapper of the same table get one copied table.
        copied = specialized_deepcopy((cache, cache, ResultCache(cache.results)))
        @test all(c -> c.results === copied[1].results, copied) && isempty(copied[1].results)
        copied[1].results[other] = [2.0]
        @test haskey(copied[3].results, other) && !haskey(cache.results, other)
    end
end

abspath(PROGRAM_FILE) == (@__FILE__) && test_result_cache()
