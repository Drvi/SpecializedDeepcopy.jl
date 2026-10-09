# Explicit graph shapes and extension contracts for the fuzz runs. Random layouts rarely build these
# shapes, and Fuzz.check cannot judge every contract: a method can reset a cache, share a table or
# reject a pointer. Each case builds its value from an rng, so the stress runner varies sizes, values
# and sharing in each round, and the test suite runs fixed seeds. Each case copies its value in two
# root orders and with a repeated root. The trim probes do not load this file.
module FuzzCases

using ..Fuzz: Fuzz
using SpecializedDeepcopy: SpecializedDeepcopy, PointerCopyError, specialized_deepcopy
using FunctionWrappers: FunctionWrapper
using LinearAlgebra: cholesky, ldiv!
using PreallocationTools: DiffCache, get_tmp
using SharedArrays: SharedArray
using SparseArrays: SparseArrays, sparse

# The README's examples of caches that share a result table, and of a callable with state.
include(joinpath(@__DIR__, "examples", "result_cache.jl"))
include(joinpath(@__DIR__, "examples", "wrapped_mutable_callable.jl"))

# A SharedArray element that holds a raw pointer. The copy does not own the target, so the method
# gives it a null pointer.
struct Address
    pointer::Ptr{Cvoid}
end
SpecializedDeepcopy.specialized_deepcopy(x::Address, copies) = Address(C_NULL)

"""
    Case(name, run)

An explicit case. `run()` copies the value of the case and returns its failed checks as strings.
"""
struct Case
    name::String
    run::Function
end

note!(reports, stage, message) = (message === nothing || push!(reports, string(stage, ": ", message)); nothing)

# Copy `value`, and copy the copy. Check both copies with Fuzz.check when `check` is true, and with
# `contract(original, copied)`, which returns a message for a failed check. With `base`, check a copy
# that Base.deepcopy makes too. Return the first copy.
function checked!(reports, value; base = true, check = true, contract = (x, y) -> nothing)
    copied = specialized_deepcopy(value)
    again = specialized_deepcopy(copied)
    for (stage, x, y) in (("copy", value, copied), ("copy of the copy", copied, again))
        check && note!(reports, stage, Fuzz.check(x, y))
        note!(reports, stage, contract(x, y))
    end
    if base
        b = deepcopy(value)
        check && note!(reports, "Base.deepcopy", Fuzz.check(value, b; bits_fields = false))
        note!(reports, "Base.deepcopy", contract(value, b))
    end
    return copied
end

name(rng) = String(rand(rng, 'a':'z', rand(rng, 0:3)))

# Keys in one table that refers to them. Each LoopKey hashes by the identity of its owner, which the
# copy sets after the copied table receives the key. The two AliasKeys compare equal for a moment of
# the copy that starts from the key with one vector. Base.deepcopy reads unset fields and crashes.
function cyclic_keys(rng)
    table = Dict{Fuzz.LoopKey, Int}()
    loop_keys = Fuzz.LoopKey[]
    for i in 1:rand(rng, 1:5)
        owner = i > 1 && rand(rng, Bool) ? loop_keys[1].owner : rand(rng, Int, rand(rng, 0:2))
        push!(loop_keys, Fuzz.LoopKey(i, table, owner))
        table[loop_keys[i]] = i
    end
    aliases = Dict{Fuzz.AliasKey, Int}()
    vector = rand(rng, Int, rand(rng, 0:2))
    alias_keys = [Fuzz.AliasKey(1, vector, aliases, vector), Fuzz.AliasKey(2, rand(rng, Int, 1), aliases, rand(rng, Int, 1))]
    for (i, k) in pairs(alias_keys)
        aliases[k] = i
    end
    return Case("cyclic keys", function ()
        reports = String[]
        for value in ((loop_keys..., table), (table, loop_keys...), (loop_keys[1], table, loop_keys[1]),
                (alias_keys..., aliases), (aliases, alias_keys...))
            checked!(reports, value; base = false)
        end
        return reports
    end)
end

# Keys that hash by their contents, with owners that values and another root share.
function shared_owners(rng)
    owners = [rand(rng, Int, rand(rng, 0:2)) for _ in 1:rand(rng, 1:3)]
    table = Dict{Fuzz.ContentKey, Any}()
    for i in 1:rand(rng, 1:5)
        owner = rand(rng, owners)
        table[Fuzz.ContentKey(i, owner)] = rand(rng, Bool) ? owner : (owner, i)
    end
    return Case("keys that share owners with values and roots", function ()
        reports = String[]
        for value in ((table, owners), (owners, table), (table, table))
            checked!(reports, value)
        end
        return reports
    end)
end

# Tuple and named tuple keys whose copy method changes them, also with a Union key type. Some
# entries share their value vector.
function tagged_keys(rng)
    n = rand(rng, 1:5)
    vectors = [[i] for i in 1:n]
    value_of(i) = rand(rng, Bool) ? vectors[1] : vectors[i]
    tuples = Dict{Fuzz.TaggedKey, Vector{Int}}()
    named = Dict{Fuzz.NamedKey, Vector{Int}}()
    optional = Dict{Union{Nothing, Fuzz.TaggedKey}, Vector{Int}}(nothing => vectors[1])
    for i in 1:n
        tuples[(Fuzz.Tag(i), name(rng))] = value_of(i)
        named[(tag = Fuzz.Tag(i), name = name(rng))] = value_of(i)
        optional[(Fuzz.Tag(i), name(rng))] = value_of(i)
    end
    return Case("tuple keys with copy methods", function ()
        reports = String[]
        for value in ((tuples, named, optional, vectors), (vectors, optional, named, tuples), (tuples, tuples))
            checked!(reports, value)
        end
        return reports
    end)
end

# Keys that hash by identity, in tables and sets, and as values and roots.
function identity_keys(rng)
    members = [Fuzz.IdentityKey(i) for i in 1:rand(rng, 1:5)]
    ids = IdDict{Fuzz.IdentityKey, Int}(k => k.id for k in members)
    table = Dict{Fuzz.IdentityKey, Fuzz.IdentityKey}(k => rand(rng, members) for k in members)
    set = Set(members)
    idset = Base.IdSet{Fuzz.IdentityKey}(members)
    return Case("identity keys in tables and sets", function ()
        reports = String[]
        for value in ((ids, table, set, members), (members, set, table, ids), (table, table))
            checked!(reports, value)
        end
        # From Julia 1.11, Base.deepcopy keeps stale objectids in a copied IdSet; see the README.
        for value in ((idset, members), (members, idset))
            checked!(reports, value; base = VERSION < v"1.11")
        end
        return reports
    end)
end

# Two caches that share one result table. The method of ResultCache starts an empty table, so
# Fuzz.check and Base.deepcopy, which keep the results, do not apply.
function shared_result_tables(rng)
    models = [Model(i) for i in 1:rand(rng, 1:3)]
    results = WeakKeyDict{Model, Vector{Float64}}(m => rand(rng, 2) for m in models)
    caches = (ResultCache(results), ResultCache(results))
    function one_empty_table(x, y)
        tables = [c.results for c in y]
        all(t -> t === tables[1], tables) && isempty(tables[1]) && tables[1] !== x[1].results && return nothing
        return "the copied caches do not share one new, empty table"
    end
    return Case("caches that share a result table", function ()
        reports = String[]
        # The models keep the entries of the weak table alive.
        GC.@preserve models for value in ((caches..., caches[1]), (caches[2], caches[1], caches[2]))
            checked!(reports, value; base = false, check = false, contract = one_empty_table)
        end
        return reports
    end)
end

# A DiffCache with a workspace from get_tmp. Each workspace container gets one copy: a new, empty
# container, unless the copy reaches the container before a cache that holds it.
function diff_caches(rng)
    n = rand(rng, 1:4)
    cache = DiffCache(rand(rng, n))
    get_tmp(cache, 1.0im) .= 1.0im
    workspaces(c) = getfield(c, 3)
    function one_empty_container(x, y)
        workspaces(y[1]) === workspaces(y[2]) && isempty(workspaces(y[1])) || return "the copied caches do not share one empty workspace container"
        y[1].du == x[1].du && y[1].du !== x[1].du || return "the data of the cache was not copied"
        size(get_tmp(y[1], 1.0im)) == size(y[1].du) || return "the copy did not allocate its own workspace"
        return nothing
    end
    function container_first(x, y)
        workspaces(y[2]) === y[1] && length(y[1]) == length(x[1]) && y[1] !== x[1] && return nothing
        return "the cache did not get the copy of its container, with its entries"
    end
    return Case("DiffCache workspaces", function ()
        reports = String[]
        checked!(reports, (cache, cache); base = false, check = false, contract = one_empty_container)
        checked!(reports, (workspaces(cache), cache); base = false, check = false, contract = container_first)
        return reports
    end)
end

# SharedArrays on this process only. Numbers get a new segment, raw pointers raise PointerCopyError,
# and an element type with a method gets the method's value.
function shared_arrays(rng)
    n = rand(rng, 1:4)
    numbers = SharedArray{Float64}(n; pids = [1])
    numbers .= rand(rng, n)
    addresses = SharedArray{Address}(n; pids = [1])
    fill!(addresses, Address(Ptr{Cvoid}(UInt(rand(rng, 1:1000) * 16))))
    # Write to the copy, check the original, and restore the copy for the next comparison.
    function independent(x, y)
        y[1] === y[2] && y[1] == x[1] && y[1].id != x[1].id || return "the copy is not one new SharedArray with the same numbers"
        before = x[1][1]
        y[1][1] = before + 1.0
        unchanged = x[1][1] == before
        y[1][1] = before
        return unchanged ? nothing : "a write to the copy changed the original"
    end
    nulled(x, y) = all(a -> a.pointer == C_NULL, y[1]) && y[1] === y[2] ? nothing : "the method of the element type did not run"
    return Case("SharedArray elements", function ()
        reports = String[]
        checked!(reports, (numbers, numbers); check = false, contract = independent)
        checked!(reports, (addresses, addresses); base = false, check = false, contract = nulled)
        storage = rand(rng, 2)
        GC.@preserve storage begin
            pointers = SharedArray{Ptr{Float64}}(1; pids = [1])
            pointers[1] = pointer(storage)
            try
                specialized_deepcopy(pointers)
                push!(reports, "copy: a SharedArray of raw pointers was copied")
            catch e
                e isa PointerCopyError || rethrow()
            end
        end
        return reports
    end)
end

# A callable with state, called or not before the copy. Calls on one copy change only its state.
function stateful_wrappers(rng)
    gain = CountingGain(rand(rng, 1:5), 0)
    wrapper = FunctionWrapper{Real, Tuple{Real}}(gain)
    rand(rng, Bool) && wrapper(1.0)
    return Case("callable with state in a FunctionWrapper", function ()
        reports = String[]
        checked!(reports, (wrapper, wrapper))
        calls = gain.calls
        first_copy, second_copy = specialized_deepcopy(wrapper), specialized_deepcopy(wrapper)
        GC.gc()
        first_copy(1.0) == gain.k || push!(reports, "call: the copy gives a wrong result")
        first_copy(1.0)
        counts = (gain.calls, first_copy.obj[].calls, second_copy.obj[].calls)
        counts == (calls, calls + 2, calls) || push!(reports, "call: the counts of the original and the copies are $counts, not $((calls, calls + 2, calls))")
        return reports
    end)
end

# A CHOLMOD workspace with buffers from a solve: both implementations give a repeated workspace one
# new, empty copy, which solves with buffers of its own.
function cholmod_workspaces(rng)
    a = rand(rng, 0.1:0.1:1.0)
    matrix = sparse([4.0 a; a 3.0])
    factor = cholesky(matrix)
    rhs = rand(rng, 2)
    workspace = SparseArrays.CHOLMOD.CholmodWS(factor)
    ldiv!(zeros(2), factor, rhs; workspace)
    buffers(w) = (w.Y[], w.E[])
    function fresh(x, y)
        y[1] === y[2] && y[1] !== x[1] && all(==(C_NULL), buffers(y[1])) || return "the copy is not one new, empty workspace"
        solution = zeros(2)
        ldiv!(solution, factor, rhs; workspace = y[1])
        solution ≈ matrix \ rhs || return "a solve with the copy gives a wrong result"
        all(((o, c),) -> o == C_NULL || c != o, zip(buffers(x[1]), buffers(y[1]))) || return "the copy uses the buffers of the original"
        return nothing
    end
    return Case("CHOLMOD workspace", function ()
        reports = String[]
        checked!(reports, (workspace, workspace); check = false, contract = fresh)
        return reports
    end)
end

"""
    cases(rng) -> Vector{Case}

The explicit cases, with values from `rng`.
"""
function cases(rng)
    all_cases = Case[cyclic_keys(rng), shared_owners(rng), tagged_keys(rng), identity_keys(rng),
        shared_result_tables(rng), diff_caches(rng), shared_arrays(rng), stateful_wrappers(rng)]
    isdefined(SparseArrays.CHOLMOD, :CholmodWS) && push!(all_cases, cholmod_workspaces(rng))
    return all_cases
end

end # module
