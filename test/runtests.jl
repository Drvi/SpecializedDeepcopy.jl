using SpecializedDeepcopy
using BlockArrays: BlockArrays, BlockedArray   # types.jl defines a `Block` of its own
using FunctionWrappers: FunctionWrapper
using LinearAlgebra: Diagonal, Symmetric, cholesky, ldiv!, transpose
using OffsetArrays: OffsetArray
using PreallocationTools: DiffCache, FixedSizeDiffCache, get_tmp
using Random: Xoshiro
using SharedArrays: SharedArray, sdata
using SparseArrays: SparseArrays, nonzeros, sparse
using Test

# `Pkg.test(; test_args = ["hammer", ...])` selects the stress runner in scripts/fuzz_hammer.jl.
if !isempty(ARGS) && ARGS[1] == "hammer"
    include(joinpath(dirname(@__DIR__), "scripts", "fuzz_hammer.jl"))
    exit(hammer(ARGS[2:end]))
end

include("types.jl")
include(joinpath("trim", "shapes.jl")) # includes fuzz.jl
include("fuzz_cases.jl")
# The README's examples. Each file defines its types and a function that runs their tests.
include(joinpath("examples", "cyclic_key.jl"))
include(joinpath("examples", "externally_allocated_memory.jl"))
include(joinpath("examples", "wrapped_mutable_callable.jl"))
include(joinpath("examples", "result_cache.jl"))

# Compare the plain family with Base.deepcopy; test custom overrides in a separate family.
const FUZZ_FAMILY = Fuzz.define_types!(Xoshiro(1), 12)
const FUZZ_METHOD_FAMILY = Fuzz.define_types!(Xoshiro(3), 12; methods = true)

# The stress runner of scripts/fuzz_hammer.jl, with the Fuzz module of this process.
module HammerTool
import ..Fuzz, ..FuzzCases
include(joinpath(dirname(@__DIR__), "scripts", "fuzz_hammer.jl"))
end

# The stress runner with stand-ins for the fixtures: no generated types, and two explicit cases. The
# first case reports a failed check. The second keeps what the file LOG holds in SEEN, and then Ctrl-C
# stops it.
module StubbedHammerTool
module Fuzz
define_types!(rng, n; kwargs...) = DataType[]
end
module FuzzCases
const LOG, SEEN = Ref(""), Ref("")
cases(_) = [(name = "failed check", run = () -> ["copy: wrong value"]),
    (name = "stopped", run = () -> (SEEN[] = read(LOG[], String); throw(InterruptException())))]
end
include(joinpath(dirname(@__DIR__), "scripts", "fuzz_hammer.jl"))
end

# A wrong-type override, also reachable through FunctionWrapper.
mutable struct WrongType
    id::Int
end
SpecializedDeepcopy.specialized_deepcopy(x::WrongType, copies) = Float64(x.id)
(x::WrongType)(n::Int) = n * x.id

# Wrong-type methods for both copies, at the top level, in a field and in an array.
mutable struct Misfit
    id::Int
end
struct MisfitHolder
    misfit::Misfit
end
SpecializedDeepcopy.specialized_deepcopy(x::Misfit, copies) = Float64(x.id)
Base.deepcopy_internal(x::Misfit, ::IdDict) = Float64(x.id)

# A vector whose method returns a Vector{Int}, which still fits a field of type AbstractVector{Int}.
struct WrongVector <: AbstractVector{Int}
    values::Vector{Int}
end
Base.size(x::WrongVector) = size(x.values)
Base.getindex(x::WrongVector, i::Int) = x.values[i]
SpecializedDeepcopy.specialized_deepcopy(x::WrongVector, copies) = copy(x.values)

# Plain values must bypass this override.
struct PlainId
    id::Int
end
SpecializedDeepcopy.specialized_deepcopy(x::PlainId, copies) = error("a method was called for a plain value")
(x::PlainId)(n::Int) = n * x.id

# Callables that hold another wrapper, as a composed model function can: one stored inline in its Ref,
# and one mutable.
struct InlineTwice
    inner::FunctionWrapper{Float64, Tuple{Float64}}
end
mutable struct MutableTwice
    inner::FunctionWrapper{Float64, Tuple{Float64}}
end
(t::Union{InlineTwice, MutableTwice})(x::Float64) = 2 * t.inner(x)

# A callable that refers to its own FunctionWrapper.
mutable struct SelfAwareLaw
    wrapper::Any
end
(::SelfAwareLaw)(x::Float64) = x

# A callable used directly as its conversion root, without a Ref.
mutable struct OwnRootLaw
    gain::Int
end
(law::OwnRootLaw)(n::Int) = law.gain * n
Base.cconvert(::Type{Ref{OwnRootLaw}}, law::OwnRootLaw) = law
Base.unsafe_convert(::Type{Ref{OwnRootLaw}}, law::OwnRootLaw) = Ptr{OwnRootLaw}(pointer_from_objref(law))

# A custom conversion root that can refer back to its wrapper.
struct RootedLaw
    gain::Int
end
(law::RootedLaw)(n::Int) = law.gain * n
mutable struct ConversionRoot
    wrapper::Any
    value::Base.RefValue{RootedLaw}
end
Base.cconvert(::Type{Ref{RootedLaw}}, law::RootedLaw) = ConversionRoot(nothing, Ref(law))
Base.unsafe_convert(::Type{Ref{RootedLaw}}, root::ConversionRoot) = Base.unsafe_convert(Ref{RootedLaw}, root.value)

# Array-backed conversion roots form cycles through mutable or immutable callables.
mutable struct ArrayRootLaw
    gain::Int
    back::Any
end
(law::ArrayRootLaw)(n::Int) = law.gain * n
Base.cconvert(::Type{Ref{ArrayRootLaw}}, law::ArrayRootLaw) = Ref([law], 1)
struct ImmutableArrayRootLaw
    gain::Int
    back::Base.RefValue{Any}
end
(law::ImmutableArrayRootLaw)(n::Int) = law.gain * n
Base.cconvert(::Type{Ref{ImmutableArrayRootLaw}}, law::ImmutableArrayRootLaw) = Ref([law], 1)

# An indirect root: wrapper -> root -> mutable storage -> Ref -> callable.
struct StoredLaw
    gain::Int
end
(law::StoredLaw)(n::Int) = law.gain * n
mutable struct LawStorage
    wrapper::Any
    value::Base.RefValue{StoredLaw}
end
mutable struct LawRoot
    storage::LawStorage
end
Base.cconvert(::Type{Ref{StoredLaw}}, law::StoredLaw) = LawRoot(LawStorage(nothing, Ref(law)))
Base.unsafe_convert(::Type{Ref{StoredLaw}}, root::LawRoot) = Base.unsafe_convert(Ref{StoredLaw}, root.storage.value)

# Plain conversion roots must bypass overrides, just like plain fields.
struct StaticLaw end
(::StaticLaw)(n::Int) = 2n
struct PlainRoot
    slot::Int
end
const STATIC_ROOTS = [Ref(StaticLaw())]
Base.cconvert(::Type{Ref{StaticLaw}}, ::StaticLaw) = PlainRoot(1)
Base.unsafe_convert(::Type{Ref{StaticLaw}}, root::PlainRoot) = Base.unsafe_convert(Ref{StaticLaw}, STATIC_ROOTS[root.slot])
SpecializedDeepcopy.specialized_deepcopy(::PlainRoot, copies) = error("a method was called for a plain root")

# A callable that overrides conversion of Base.cconvert's Ref.
struct PtrLaw
    gain::Int
end
(law::PtrLaw)(n::Int) = law.gain * n
Base.unsafe_convert(::Type{Ptr{PtrLaw}}, r::Base.RefValue{PtrLaw}) = Ptr{PtrLaw}(pointer_from_objref(r))

# An immutable callable that its Ref stores boxed, because a constructor can leave a field unset.
mutable struct LawGain
    wrapper::Any
    gain::Int
end
struct BoxedLaw
    state::LawGain
    extra::Vector{Int}
    BoxedLaw(state) = new(state)
    BoxedLaw(state, extra) = new(state, extra)
end
(law::BoxedLaw)(n::Int) = law.state.gain * n

# Reserve this type family for the inference test to avoid earlier cached specializations.
mutable struct Layer{T}
    value::T
end
@generated nested_layers(x, ::Val{N}) where {N} = foldl((ex, _) -> :(Layer($ex)), 1:N; init = :x)

# Two closures that capture one reassigned variable, so they share one Core.Box.
function shared_counter()
    count = 0
    add() = (count += 1)
    read() = count
    return add, read
end

# A local function that calls itself captures itself through a Core.Box.
function recursive_closure()
    fib = n -> n < 2 ? n : fib(n - 1) + fib(n - 2)
    return fib
end

# Tuple and named tuple keys that hold a type of their own, with methods that change them.
struct KeyTag
    id::Int
end
SpecializedDeepcopy.specialized_deepcopy(k::Tuple{KeyTag, String}, copies) = (k[1], "copied-" * k[2])
SpecializedDeepcopy.specialized_deepcopy(k::@NamedTuple{tag::KeyTag, name::String}, copies) = (tag = k.tag, name = "copied-" * k.name)

# An immutable key type that refers to itself.
struct KeyChain
    name::String
    next::Union{Nothing, KeyChain}
end

# A key that reaches its table through other objects, with a hash that reads a field of one of them.
mutable struct Circuit
    pins::Any
    owner::Vector{Int}
end
struct Pin
    circuit::Circuit
end
Base.hash(p::Pin, h::UInt) = hash(objectid(p.circuit.owner), h)

# A key that holds the address of its owner, which a registry keeps alive. Its hash reads a field of
# the owner, and its copy method copies the owner.
struct OwnerAddress
    address::Ptr{Cvoid}
end
mutable struct AddressOwner
    table::Dict{OwnerAddress, Int}
    storage::Vector{Int}
end
const ADDRESS_OWNERS = Dict{Ptr{Cvoid}, AddressOwner}()
function owner_address(owner::AddressOwner)
    address = pointer_from_objref(owner)
    ADDRESS_OWNERS[address] = owner
    return OwnerAddress(address)
end
Base.hash(k::OwnerAddress, h::UInt) = hash(objectid(ADDRESS_OWNERS[k.address].storage), h)
SpecializedDeepcopy.specialized_deepcopy(k::OwnerAddress, copies) = owner_address(specialized_deepcopy(ADDRESS_OWNERS[k.address], copies))

# Two tables whose keys depend on the copy of a table: two keys of `table` compare equal for a moment
# of the copy, and the hash of a SizeKey reads the length of its table.
mutable struct PairKey
    left::Vector{Int}
    table::Dict{PairKey, Any}
    right::Vector{Int}
end
Base.hash(k::PairKey, h::UInt) = hash(k.left === k.right, h)
Base.isequal(a::PairKey, b::PairKey) = (a.left === a.right) == (b.left === b.right)
struct SizeKey
    table::Dict{PairKey, Any}
    id::Int
end
Base.hash(k::SizeKey, h::UInt) = hash(length(k.table), hash(k.id, h))

# Keys of two tables whose equality reads the length of the other table, a cycle between the tables.
mutable struct PeerKey
    id::Int
    peer::Any
end
peer_class(k::PeerKey) = min(k.id, length(k.peer) + 1)
Base.hash(k::PeerKey, h::UInt) = hash(peer_class(k), h)
Base.isequal(a::PeerKey, b::PeerKey) = peer_class(a) == peer_class(b)

# Run code in a fresh Julia process with this test project and capture stdout.
run_isolated(code, flags = ``) =
    read(`$(Base.julia_cmd()) --startup-file=no $flags --project=$(Base.active_project()) -e $code`, String)

# Run code that can fail or crash in a fresh Julia process. Return whether the process succeeded,
# and its stdout. The stderr of a process that fails, such as a crash report, goes to the test log.
function run_outcome(code)
    output, errors = IOBuffer(), IOBuffer()
    command = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) -e $code`
    finished = success(pipeline(command; stdout = output, stderr = errors))
    finished || print(stderr, String(take!(errors)))
    return finished, String(take!(output))
end

@testset "SpecializedDeepcopy" begin
    @testset "method table" begin
        @test isempty(Test.detect_ambiguities(SpecializedDeepcopy))
    end

    @testset "aliases and cycles" begin
        owner = Owner(0.0)
        buffer = [owner, owner]
        numeric = [1.0, 2.0]
        a, b, c, d = specialized_deepcopy((buffer, buffer, [owner], [numeric, numeric]))
        @test a === b !== buffer
        @test a[1] === a[2] === c[1] !== owner
        @test d[1] === d[2] !== numeric
        @test d[1] == numeric
        a[1].id = 1.0
        d[1][1] = 3.0
        @test owner.id == 0.0 && numeric == [1.0, 2.0]
        fresh = specialized_deepcopy(buffer)
        @test fresh[1].id == 0.0 && fresh[1] !== a[1]

        node = Node("loop", nothing)
        node.next = node
        copied = specialized_deepcopy(node)
        @test copied !== node && copied.next === copied && copied.label == "loop"

        cyclic = Any[]
        push!(cyclic, cyclic)
        copied = specialized_deepcopy(cyclic)
        @test copied !== cyclic && copied[1] === copied

        group = Group(Group[])
        push!(group.members, group)
        copied = specialized_deepcopy(group)
        @test copied !== group && only(copied.members) === copied
    end

    @testset "undefined fields and elements" begin
        lazy = specialized_deepcopy(Lazy(1))
        @test lazy.id == 1 && !isdefined(lazy, :cache)
        @test !isdefined(specialized_deepcopy(Prefix(1)), :first)
        partial = specialized_deepcopy(Prefix(1, [2.0]))
        @test partial.first == [2.0] && !isdefined(partial, :second)
        full = Prefix(1, [2.0], [3.0])
        copied = specialized_deepcopy(full)
        @test copied.second == [3.0] && copied.second !== full.second
        @test !isdefined(specialized_deepcopy(LateTuple()), :parts)
        late = LateTuple((1, [2.0]))
        copied = specialized_deepcopy(late)
        @test copied.parts == late.parts && copied.parts[2] !== late.parts[2]
        @test !isdefined(specialized_deepcopy(Never()), :value)
        @test !isdefined(specialized_deepcopy(MutableNever()), :value)

        holes = Vector{Owner}(undef, 2)
        holes[1] = Owner(1.0)
        copied = specialized_deepcopy(holes)
        @test copied[1].id == 1.0 && copied[1] !== holes[1] && !isassigned(copied, 2)
        @test !isdefined(specialized_deepcopy(Core.Box()), :contents)
    end

    @testset "exact types" begin
        # Like HVACComponents' refrigerant data: declared field types are abstract, values are concrete.
        Data = NamedTuple{(:name, :PMin, :PVector), Tuple{String, AbstractFloat, AbstractVector{<:AbstractFloat}}}
        medium = Record(Data(("R32", 1.0f0, Float32[1, 2])))
        copied = specialized_deepcopy(medium)
        @test typeof(copied) === typeof(medium)
        @test copied.value.PVector == medium.value.PVector && copied.value.PVector !== medium.value.PVector

        @test typeof(specialized_deepcopy(Record{Real}(1.0))) === Record{Real}

        optional = Union{Nothing, Owner}[nothing, Owner(1.0)]
        copied = specialized_deepcopy(optional)
        @test typeof(copied) === typeof(optional) && copied[1] === nothing && copied[2].id == 1.0
    end

    @testset "const and atomic fields" begin
        owner = Owner(1.0)
        counter = Counter("hits", owner, 3)
        copied = specialized_deepcopy(counter)
        @test copied.name == "hits" && copied.owner !== owner && copied.owner.id == 1.0
        @test (@atomic copied.count) == 3
        @atomic copied.count += 1
        @test (@atomic counter.count) == 3
    end

    @testset "arrays" begin
        matrix = reshape(collect(1.0:6.0), 2, 3)
        copied = specialized_deepcopy(matrix)
        @test copied == matrix && copied !== matrix

        bits = BitVector([true, false])
        copied, again = specialized_deepcopy((bits, bits))
        @test copied === again !== bits && copied == bits
        copied[1] = false
        @test bits[1]

        # Match Base's reshape sharing: separate buffers on 1.10, shared Memory from 1.11.
        shared = isdefined(Base, :Memory)
        flat = collect(1.0:4.0)
        square = reshape(flat, 2, 2)
        same_shape = reshape(square, 4)
        copied_flat, copied_square, copied_same_shape = specialized_deepcopy((flat, square, same_shape))
        base_flat, base_square, base_same_shape = deepcopy((flat, square, same_shape))
        copied_flat[1] = 10.0
        base_flat[1] = 10.0
        @test flat[1] == 1.0 && copied_same_shape !== copied_flat
        @test (copied_square[1, 1] == 10.0) == (base_square[1, 1] == 10.0) == shared
        @test (copied_same_shape[1] == 10.0) == (base_same_shape[1] == 10.0) == shared
        grows(v) = try
            push!(v, 0.0)
            true
        catch
            false
        end
        @test grows(copied_flat) == grows(base_flat)
        owners = [Owner(1.0), Owner(2.0)]
        copied_grid, copied_owners = specialized_deepcopy((reshape(owners, 1, 2), owners))
        copied_owners[2] = Owner(5.0)
        @test copied_grid[1, 1] === copied_owners[1] !== owners[1]
        @test (copied_grid[1, 2] === copied_owners[2]) == shared

        owners = [Owner(1.0), Owner(2.0)]
        copied_owners, copied_view = specialized_deepcopy((owners, view(owners, 2:2)))
        @test copied_view[1] === copied_owners[2] !== owners[2]

        shifted = OffsetArray([Owner(1.0), Owner(2.0)], -1)
        copied = specialized_deepcopy(shifted)
        @test axes(copied) == axes(shifted)
        @test copied[0].id == 1.0 && copied[0] !== shifted[0]

        # popfirst! moves the start of the vector inside its storage.
        moved = [Owner(1.0), Owner(2.0), Owner(3.0)]
        popfirst!(moved)
        copied = specialized_deepcopy(moved)
        @test length(copied) == 2 && copied[1].id == 2.0 && copied[1] !== moved[1]

        zero_dimensional = fill(Owner(1.0))
        @test specialized_deepcopy(zero_dimensional)[].id == 1.0

        # An empty comprehension has the element type Union{}.
        no_elements = [x for x in ()]
        @test eltype(no_elements) === Union{} && typeof(specialized_deepcopy(no_elements)) === Vector{Union{}}
        @test specialized_deepcopy(no_elements) !== no_elements && length(specialized_deepcopy(Vector{Union{}}(undef, 2))) == 2
    end

    @testset "hash tables" begin
        owner = Owner(1.0)
        table = Dict(owner => [1.0])
        copied = specialized_deepcopy(table)
        key = only(keys(copied))
        @test key !== owner && copied[key] == [1.0] && copied[key] !== table[owner]

        copied_owners, copied_table = specialized_deepcopy(([owner], Dict(owner => 1)))
        @test copied_table[only(copied_owners)] == 1

        ids = specialized_deepcopy(IdDict(owner => 2))
        @test ids[only(keys(ids))] == 2 && !haskey(ids, owner)

        set = specialized_deepcopy(Base.IdSet{Owner}([owner]))
        @test only(set) in set && !(owner in set)

        copied_set = specialized_deepcopy(Set([owner]))
        @test only(copied_set) in copied_set

        plain = Dict("gain" => 2.0)
        copied = specialized_deepcopy(plain)
        @test copied == plain && copied !== plain

        # Tables with String, Symbol, plain or Union keys keep their slots, and tables with tuple keys
        # that hold a String are rebuilt. Either way, lookups by the original keys work, values are
        # copied, and a deleted entry stays deleted.
        stable = (
            Dict("key$i" => [Float64(i)] for i in 1:40),
            Dict(("key$i", i) => [i] for i in 1:40),
            Dict((name = "key$i", id = i) => [i] for i in 1:40),
            Dict{Union{Int, String}, Vector{Int}}((isodd(i) ? i : "key$i") => [i] for i in 1:40),
            Dict{Union{Nothing, Symbol}, Vector{Int}}(nothing => [0], :a => [1]),
            Set("key$i" for i in 1:40),
        )
        for table in stable
            delete!(table, first(table isa AbstractSet ? table : keys(table)))
            copied = specialized_deepcopy(table)
            @test typeof(copied) === typeof(table) && length(copied) == length(table) && copied == table
            if table isa AbstractDict
                @test all(haskey(copied, k) && copied[k] == v && copied[k] !== v for (k, v) in table)
                @test all(haskey(table, k) for k in keys(copied))
            else
                @test all(k in copied for k in table) && all(k in table for k in copied)
            end
        end
        by_symbol = Dict(Symbol("key$i") => Owner(i) for i in 1:40)
        copied = specialized_deepcopy(by_symbol)
        @test all(copied[k].id == v.id && copied[k] !== v for (k, v) in by_symbol)
        shared = [1.0]
        copied = specialized_deepcopy(Dict("a" => shared, "b" => shared))
        @test copied["a"] === copied["b"] !== shared
        cyclic = Dict{String, Any}()
        cyclic["self"] = cyclic
        copied = specialized_deepcopy(cyclic)
        @test copied["self"] === copied !== cyclic
        # A table with a mutable key among its keys is rebuilt, so the copied key is found.
        mixed = Dict{Any, Any}(owner => 1, "s" => 2, 3 => 3)
        copied = specialized_deepcopy(mixed)
        @test copied[only(k for k in keys(copied) if k isa Owner)] == 1 && copied["s"] == 2 && !haskey(copied, owner)
        # A method can change a tuple or named tuple key that holds a type of its own, so its table is
        # rebuilt, also when the declared key type is a Union.
        tagged = [(KeyTag(i), "key$i") for i in 1:40]
        named = [(tag = KeyTag(i), name = "key$i") for i in 1:40]
        for table in (Dict(k => [i] for (i, k) in pairs(tagged)), Dict(k => [i] for (i, k) in pairs(named)),
                Dict{Union{Nothing, Tuple{KeyTag, String}}, Vector{Int}}(k => [i] for (i, k) in pairs(tagged)))
            copied = specialized_deepcopy(table)
            @test length(copied) == 40 && all(k -> startswith(k[2], "copied-") && haskey(copied, k), keys(copied))
        end
        # The hash of this key reads a field that the copy sets after the table receives the key. The
        # check after the copy reinserts the entry. test/examples/cyclic_key.jl has the direct case.
        circuit = Circuit(nothing, [1])
        circuit.pins = (Dict(Pin(circuit) => 1),)
        copied = specialized_deepcopy(circuit)
        @test copied.pins[1][Pin(copied)] == 1 && copied.owner !== circuit.owner
        # Two keys whose equality reads whether `left` and `right` are one vector compare equal for a
        # moment of the copy that starts from the key with one vector. The table keeps both entries in
        # either root order.
        shared = [0]
        aliases = Dict{Fuzz.AliasKey, Int}()
        alias_keys = (Fuzz.AliasKey(1, shared, aliases, shared), Fuzz.AliasKey(2, [1], aliases, [2]))
        aliases[alias_keys[1]], aliases[alias_keys[2]] = 1, 2
        for value in (alias_keys, (aliases, alias_keys...))
            copied = specialized_deepcopy(value)
            @test Fuzz.check(value, copied) === nothing && length(copied[end].table) == 2
        end
        # A key that holds a raw pointer can lead to objects that the copy sets after the table receives
        # the key. Each owner reaches its own table first, so the hash of each copied key changes after
        # the insertion. With eight tables, lookups that work by chance are unlikely.
        address_owners = [AddressOwner(Dict{OwnerAddress, Int}(), [i]) for i in 1:8]
        for (i, owner) in pairs(address_owners)
            owner.table[owner_address(owner)] = i
        end
        copied = specialized_deepcopy(address_owners)
        @test all(pairs(copied)) do (i, owner)
            key = only(keys(owner.table))
            ADDRESS_OWNERS[key.address] === owner && get(owner.table, key, 0) == i &&
                owner.storage == [i] && owner.storage !== address_owners[i].storage
        end
        # Refilling `pair_table` changes the hash of the key of `sizes`, so the check needs a second pass.
        pair_table = Dict{PairKey, Any}()
        vector = [1]
        pair_keys = (PairKey(vector, pair_table, vector), PairKey([2], pair_table, [3]))
        pair_table[pair_keys[1]], pair_table[pair_keys[2]] = 1, 2
        sizes = Dict(SizeKey(pair_table, 1) => 42)
        pair_table[pair_keys[1]], pair_table[pair_keys[2]] = sizes, sizes
        first_key, second_key, copied_table, copied_sizes = specialized_deepcopy((pair_keys..., pair_table, sizes))
        @test length(copied_table) == 2 && copied_table[first_key] === copied_table[second_key] === copied_sizes
        @test get(copied_sizes, SizeKey(copied_table, 1), 0) == 42
        @test first_key.table === copied_table === second_key.table
        @test first_key.left === first_key.right && second_key.left !== second_key.right
        # Keys whose equality reads the length of the other table form a cycle between two tables. With
        # six entries per table, the tables need more passes than the limit of one more than the number
        # of tables, so the copy raises an error.
        peers = [Dict(PeerKey(i, zeros(Int, 6)) => i for i in 1:6) for _ in 1:2]
        foreach(k -> k.peer = peers[2], keys(peers[1]))
        foreach(k -> k.peer = peers[1], keys(peers[2]))
        @test_throws "in all 3 passes" specialized_deepcopy(peers)
        # Two copied keys that are equal after the copy cannot both stay: an error, not a lost entry.
        copies = IdDict{Any, Any}()
        SpecializedDeepcopy._check_later!(copies, Dict{Vector{Int}, Int}(), [[1] => 1, [1] => 2])
        @test_throws ArgumentError SpecializedDeepcopy._check_tables!(copies)
        # Only a table whose keys can change after the table receives them gets that check.
        settled = (String, Tuple{String, Int}, @NamedTuple{tag::KeyTag, name::String}, Union{Nothing, Symbol}, KeyChain)
        @test all(SpecializedDeepcopy._settled_key, settled)
        @test !any(SpecializedDeepcopy._settled_key, (Owner, Pin, Tuple{Owner, Int}, Any, Vector{Int}, OwnerAddress, Union{Nothing, Ptr{Cvoid}}))
    end

    @testset "Base types" begin
        text = "text"
        @test specialized_deepcopy(text) == text
        @test specialized_deepcopy(:symbol) === :symbol
        @test specialized_deepcopy(Vector{Int}) === Vector{Int}
        regex = r"a+"
        @test specialized_deepcopy(regex) === regex
        task = Task(() -> nothing)
        @test specialized_deepcopy(task) === task

        large = big(2)^100
        copied = specialized_deepcopy(large)
        @test copied == large && copied !== large
        precise = BigFloat(1.5; precision = 300)
        copied = specialized_deepcopy(precise)
        @test copied == precise && copied !== precise && precision(copied) == 300

        @test_throws ErrorException specialized_deepcopy(Base)

        held = ReentrantLock()
        lock(held)
        copied = specialized_deepcopy(held)
        @test copied !== held && !islocked(copied)
        unlock(held)
        condition = Threads.Condition()
        @test specialized_deepcopy(condition).lock !== condition.lock
        shared = SharedLock()
        for value in ((shared, Base.GenericCondition(shared)), (Base.GenericCondition(shared), shared))
            copied = specialized_deepcopy(value)
            copied_condition = value[1] isa SharedLock ? copied[2] : copied[1]
            @test copied_condition.lock === shared && copied_condition !== value[1] && copied_condition !== value[2]
        end
        # Repeated conditions must retain one wait queue.
        first_copy, second_copy = specialized_deepcopy((condition, condition))
        @test first_copy === second_copy && first_copy.waitq === second_copy.waitq
        # Follow the cycle condition -> lock -> condition.
        back = BackLock(nothing)
        back.condition = Base.GenericCondition(back)
        copied = specialized_deepcopy(back.condition)
        @test copied.lock.condition === copied && copied.lock !== back

        state = [0]
        bump = () -> (state[1] += 1)
        specialized_deepcopy(bump)()
        @test state == [0]

        increment = boxed_counter()
        increment()
        copied = specialized_deepcopy(increment)
        @test copied() == 2
        @test increment() == 2
    end

    @testset "raw pointers" begin
        address = Ptr{Cvoid}(UInt(0x1000))
        @test_throws PointerCopyError(Handle, :address) specialized_deepcopy(Handle(address))
        @test_throws PointerCopyError(Handle, :address) specialized_deepcopy(Device(Handle(address), 1.0))
        @test_throws PointerCopyError(Handle, :address) specialized_deepcopy([Handle(address)])
        @test_throws PointerCopyError(RawBuffer, :data) specialized_deepcopy(RawBuffer(Ptr{UInt8}(C_NULL)))
        @test_throws PointerCopyError(Name, :text) specialized_deepcopy(Name(reinterpret(Cstring, C_NULL)))
        @test_throws PointerCopyError(Ptr{Cvoid}, nothing) specialized_deepcopy(address)
        @test_throws PointerCopyError(Ptr{Cvoid}, nothing) specialized_deepcopy(Any[address])
        @test_throws PointerCopyError(Tuple{Ptr{Cvoid}, Int}, 1) specialized_deepcopy((address, 1))
        @test specialized_deepcopy(MaybePointer(nothing)).address === nothing
        @test_throws PointerCopyError(Ptr{Cvoid}, nothing) specialized_deepcopy(MaybePointer(address))
        # A struct that nests into itself through a Union with a pointer member, with a const field in
        # the mutable form: a chain that ends with `nothing` copies, one that ends with a pointer throws.
        levels(x) = x === nothing ? 0 : 1 + levels(x.t)
        for T in (Tricky, ImmutableTricky)
            chain = tricky_chain(T, nothing, Val(4))
            copied = specialized_deepcopy(chain)
            @test typeof(copied) === T && levels(copied) == 4 && copied.t.t.ref[] == 2
            @test copied.ref === copied.cref !== chain.ref && copied.t.ref === copied.t.cref !== chain.t.ref
            copied.t.cref[] = 20
            @test chain.t.cref[] == 3
            @test_throws PointerCopyError(Ptr{UInt}, nothing) specialized_deepcopy(tricky_chain(T, Ptr{UInt}(UInt(0x10)), Val(4)))
        end
        # The same through two types that refer to each other, also when the chain closes into a cycle.
        chain = ping_chain(3, nothing)
        copied = specialized_deepcopy(chain)
        @test copied.id == 3 && copied.next.back.id == 2 && copied.next.back.next.back.next === nothing
        @test copied.next !== chain.next && copied.next.back.next.back !== chain.next.back.next.back
        @test_throws PointerCopyError(Ptr{Cvoid}, nothing) specialized_deepcopy(ping_chain(3, address))
        @test_throws PointerCopyError(Cstring, nothing) specialized_deepcopy(ping_chain(3, Pong(reinterpret(Cstring, C_NULL))))
        chain.next.back.next.back.next = Pong(chain)
        copied = specialized_deepcopy(chain)
        @test copied.next.back.next.back.next.back === copied !== chain
        # A Union with a pointer member is also split as an element type.
        maybe = Union{Nothing, Ptr{Cvoid}}[nothing, nothing]
        @test specialized_deepcopy(maybe) == maybe && typeof(specialized_deepcopy(maybe)) === typeof(maybe)
        maybe[2] = address
        @test_throws PointerCopyError(Ptr{Cvoid}, nothing) specialized_deepcopy(maybe)
        @test_throws PointerCopyError(WeakRef, nothing) specialized_deepcopy(WeakRef(Owner(1.0)))
        mktemp() do _, io
            @test_throws PointerCopyError(IOStream, :handle) specialized_deepcopy(io)
        end

        message = sprint(showerror, PointerCopyError(Handle, :address))
        @test occursin("Handle", message) && occursin("`address`", message) && occursin("Ptr{Nothing}", message)
        # A method for a type that the caller does not define, such as IOStream, would be piracy.
        @test occursin("a type of your own", message)
        @test occursin("WeakRef", sprint(showerror, PointerCopyError(WeakRef, nothing)))

        device = SharedDevice(SharedHandle(address), 2.0)
        @test specialized_deepcopy(device) === device
        @test only(specialized_deepcopy([device])) === device

        # A table finds every key whose method gives the copy another pointer and so another hash.
        table = Dict(BytesKey(UInt8[i]) => Int(i) for i in 0x01:0x10)
        copied = specialized_deepcopy(table)
        @test length(copied) == 16 && all(k -> haskey(copied, k), keys(copied)) && isdisjoint(keys(copied), keys(table))
        @test all(k -> KEY_BYTES[k.address] == UInt8[copied[k]], keys(copied))
        @test !any(SpecializedDeepcopy._stable_hash, (BytesKey, Union{String, BytesKey}, Tuple{Int, BytesKey}))
    end

    @testset "methods for user types" begin
        @test_throws TypeError specialized_deepcopy([WrongType(1)])
        @test_throws TypeError specialized_deepcopy(WrongType(1))
        @test_throws TypeError specialized_deepcopy(Core.svec(WrongType(1)))

        plain = PlainId(1)
        @test specialized_deepcopy(plain) === plain
        @test only(specialized_deepcopy([plain])) === plain
        @test specialized_deepcopy(Record(plain)).value === plain
        @test only(specialized_deepcopy(Any[plain])) === plain
        @test only(keys(specialized_deepcopy(Dict(plain => [1])))) === plain

        # Tagged's override also runs inside containers; fields that constructors can leave unset use a bit copy.
        tag = Tagged(3)
        @test specialized_deepcopy(tag).value == 4
        @test specialized_deepcopy(TaggedHolder(tag)).tag.value == 4
        @test specialized_deepcopy((tag, 1))[1].value == 4 && specialized_deepcopy((a = tag,)).a.value == 4
        @test specialized_deepcopy(Some(tag)).value.value == 4 && only(specialized_deepcopy([tag])).value == 4
        @test specialized_deepcopy(Dict(1 => tag))[1].value == 4
        @test specialized_deepcopy(LateTagged(1, tag)).tag.value == 3
    end

    @testset "closures" begin
        # Call closures to check captured state without depending on lowered field names.
        weights = [1.0, 2.0]
        owner = Owner(1.0)
        scale = x -> weights[1] * x * owner.id
        copied_scale, copied_weights, copied_owner = specialized_deepcopy((scale, weights, owner))
        copied_weights[1] = 10.0
        copied_owner.id = 3.0
        @test copied_scale(1.0) == 30.0 && scale(1.0) == 1.0

        # A function without captures is a singleton, so the copy is the function itself.
        @test specialized_deepcopy(sin) === sin
        increment = x -> x + 1
        @test specialized_deepcopy(increment) === increment

        # Shared captured variables must remain shared through one copied Core.Box.
        add, read = shared_counter()
        add()
        copied_add, copied_read = specialized_deepcopy((add, read))
        copied_add()
        @test copied_read() == 2 && read() == 1

        # The only field of this closure is the Core.Box that holds the closure itself.
        fib = recursive_closure()
        copied = specialized_deepcopy(fib)
        @test getfield(copied, 1).contents === copied !== fib && copied(10) == 55

        outer = let inner = x -> weights[2] * x
            x -> inner(x) + 1
        end
        copied = specialized_deepcopy(outer)
        weights[2] = 100.0
        @test copied(1.0) == 3.0 && outer(1.0) == 101.0

        fixed = Base.Fix1(*, [2.0])
        copied = specialized_deepcopy(fixed)
        @test copied.x !== fixed.x && copied(3.0) == [6.0]
        copied = specialized_deepcopy(first ∘ Returns(weights))
        @test copied.inner.value !== weights && copied() == 1.0

        # A closure that captures a raw pointer needs a method for the type that holds the pointer.
        handle = Handle(Ptr{Cvoid}(UInt(0x1000)))
        @test_throws PointerCopyError(Handle, :address) specialized_deepcopy(() -> handle)
        # An opaque closure holds pointers to compiled code.
        opaque = Base.Experimental.@opaque x -> x + 1
        @test_throws PointerCopyError(typeof(opaque), :invoke) specialized_deepcopy(opaque)
    end

    # Synthetic MTKParameters.nonnumeric buffers with concrete, abstract and Any elements.
    @testset "nonnumeric parameters" begin
        buffers = nonnumeric_buffers()
        medium = only(buffers[1])
        density = buffers[2][1]
        parent, child = buffers[3]
        everything = (buffers..., AbstractTable[density, ConstantTable(1.0)], Any[medium, 1.0, "text"],
            [FunctionWrapper{Float64, Tuple{Float64}}(parent.law)])
        copied = specialized_deepcopy(everything)
        @test typeof(copied) === typeof(everything)

        # Preserve aliases across concrete, abstract and Any element types.
        copied_medium = only(copied[1])
        copied_parent, copied_child = copied[3]
        @test copied[2][1] === copied_medium.tables[:density] === copied[7][1] !== density
        @test copied[8][1] === copied_medium !== medium
        @test copied_child.parent === copied_parent !== parent && copied_parent.medium === copied_medium

        # Mutations through several copied buffers must leave the originals unchanged.
        copied_medium.tables[:density].last_index[] = 2
        copied_medium.data.coefficients[1] = 10.0
        push!(copied_parent.history, 2.0)
        copied_parent.active[1] = false
        copied[4][1]["a"][1] = 10.0
        @test density.last_index[] == 1 && medium.data.coefficients[1] == 1.0
        @test length(parent.history) == 2 && parent.active[1] && buffers[4][1]["a"] == [1.0]

        # Both controllers and the wrapper must use the same copied weights.
        copied_weights = getfield(copied_parent.law, 1)
        copied_weights[1] = 5.0
        @test copied_child.law(1.0) == 5.0 && only(copied[9])(1.0) == 5.0 && parent.law(1.0) == 1.0

        @test copied_medium.unit === :SI && copied_medium.limits === (0.0, missing)
        @test copied[2][2] === nothing && copied_child.medium === nothing
        @test isequal(copied_parent.history, [missing, 1.0, 2.0])

        # Arrays preserve a declared tuple element type containing a Union.
        pairs_ = Tuple{Float64, Union{Missing, Float64}}[(0.0, missing), (1.0, 2.0)]
        @test isequal(specialized_deepcopy(pairs_), pairs_) && typeof(specialized_deepcopy(pairs_)) === typeof(pairs_)
    end

    # The workspace field is `any_du` or `typed_du`, depending on the PreallocationTools version.
    workspaces(cache) = getfield(cache, 3)
    # A cache of the family C from its fields. From PreallocationTools 1.2, a DiffCache has a fourth
    # field, warn_on_resize. The FixedSizeDiffCache constructors need ForwardDiff, so every test cache
    # other than DiffCache(zeros(3)) is built from its fields.
    W = fieldtype(FixedSizeDiffCache, 3)
    build(C, du, dual_du, workspace = W()) =
        fieldcount(C{typeof(du), typeof(dual_du)}) == 4 ? C(du, dual_du, workspace, true) : C(du, dual_du, workspace)

    @testset "PreallocationTools caches" begin
        cache = DiffCache(zeros(3))
        get_tmp(cache, 1.0im) .= 1.0im
        @test !isempty(workspaces(cache))
        copied, du = specialized_deepcopy((cache, cache.du))
        @test typeof(copied) === typeof(cache) && copied.du === du !== cache.du
        @test copied.du == cache.du && copied.dual_du == cache.dual_du && copied.dual_du !== cache.dual_du
        @test all(i -> getfield(copied, i) == getfield(cache, i), 4:nfields(cache))
        # The copy starts without workspaces and allocates its own.
        @test isempty(workspaces(copied))
        @test size(get_tmp(copied, 1.0im)) == size(copied.du) && get_tmp(copied, 1.0im) !== get_tmp(cache, 1.0im)
        @test get_tmp(copied, 1.0) === copied.du
        copied.du[1] = 5.0
        @test cache.du[1] == 0.0
        # Repeated caches stay one value with one workspace container, as with Base.deepcopy, so a
        # table keyed by the cache finds the copy in either root order.
        pair = [cache, cache]
        copied_pair, base_pair = specialized_deepcopy(pair), deepcopy(pair)
        @test copied_pair[1] === copied_pair[2] && base_pair[1] === base_pair[2]
        @test copied_pair[1].du == base_pair[1].du && copied_pair[1].dual_du == base_pair[1].dual_du
        # get_tmp gives the copies one scratch array exactly when it gives the original one; whether it
        # does depends on the PreallocationTools version.
        @test (get_tmp(copied_pair[1], 1.0im) === get_tmp(copied_pair[2], 1.0im)) == (get_tmp(cache, 1.0im) === get_tmp(cache, 1.0im))
        table = Dict(cache => 1)
        for value in ((table, cache), (cache, table))
            copied = specialized_deepcopy(value)
            copied_table, copied_cache = value[1] isa Dict ? copied : reverse(copied)
            @test copied_table[copied_cache] == 1
        end
        # A cache that shares the dual buffer and the workspace container of another, as
        # `reshape(cache, ...)` makes one from PreallocationTools 1.3.
        shaped = build(DiffCache, reshape(cache.du, 3, 1), cache.dual_du, workspaces(cache))
        for value in ((cache, shaped), (shaped, cache))
            a, b = specialized_deepcopy(value)
            @test workspaces(a) === workspaces(b) && isempty(workspaces(a)) && a.dual_du === b.dual_du
        end
        # A container that the copy reaches before any cache that holds it keeps its workspaces.
        copied_workspaces, copied = specialized_deepcopy((workspaces(cache), cache))
        @test workspaces(copied) === copied_workspaces && length(copied_workspaces) == length(workspaces(cache))
        copied, copied_workspaces = specialized_deepcopy((cache, workspaces(cache)))
        @test workspaces(copied) === copied_workspaces && isempty(copied_workspaces)
        fixed = build(FixedSizeDiffCache, zeros(2), zeros(2))
        get_tmp(fixed, 1.0im)
        copied_fixed = specialized_deepcopy(fixed)
        @test typeof(copied_fixed) === typeof(fixed) && copied_fixed.du == fixed.du && copied_fixed.du !== fixed.du
        @test copied_fixed.dual_du !== fixed.dual_du && isempty(workspaces(copied_fixed))
        @test size(get_tmp(copied_fixed, 1.0im)) == size(fixed.du)
        copied_pair = specialized_deepcopy([fixed, fixed])
        @test copied_pair[1] === copied_pair[2]
        # Data buffers are copied, also when they have the type of a workspace container.
        for untyped in (build(DiffCache, Any[1.0, 2.0], Any[3.0, 4.0]), build(FixedSizeDiffCache, Any[1.0, 2.0], Any[3.0, 4.0]))
            copied = specialized_deepcopy(untyped)
            @test copied.du == untyped.du !== copied.du && copied.dual_du == untyped.dual_du !== copied.dual_du
        end
        # Data buffers follow the pointer rule and the methods of their types.
        memory = [3.0, 4.0]
        GC.@preserve memory begin
            borrowed, shared = BorrowedVector(pointer(memory), 2), SharedVector(pointer(memory), 2)
            @test_throws PointerCopyError(BorrowedVector, :address) specialized_deepcopy(build(DiffCache, borrowed, zeros(2)))
            @test_throws PointerCopyError(BorrowedVector, :address) specialized_deepcopy(build(FixedSizeDiffCache, borrowed, zeros(2)))
            @test specialized_deepcopy(build(DiffCache, shared, zeros(2))).du === shared
        end
        # The copy of a data buffer must have the type of the buffer, also when the declared type of
        # the field is abstract.
        for C in (DiffCache, FixedSizeDiffCache)
            T = C{AbstractVector{Int}, Vector{Int}}
            wrong = fieldcount(T) == 4 ? T(WrongVector([1, 2]), [3, 4], W(), true) : T(WrongVector([1, 2]), [3, 4], W())
            @test_throws TypeError specialized_deepcopy(wrong)
        end
    end

    # A clocked system's discrete parameter portion.
    @testset "BlockedArray" begin
        blocked = BlockedArray([1.0, 2.0, 3.0], [2, 1])
        copied = specialized_deepcopy(blocked)
        @test typeof(copied) === typeof(blocked) === typeof(deepcopy(blocked)) && copied == blocked
        @test copied.blocks !== blocked.blocks && axes(copied, 1).lasts !== axes(blocked, 1).lasts
        copied[1] = 10.0
        @test blocked[1] == 1.0 && copied[BlockArrays.Block(1)] == [10.0, 2.0] && copied[BlockArrays.Block(2)] == [3.0]
    end

    @testset "abstract arrays" begin
        owners = [Owner(i + 10j) for i in 1:3, j in 1:3]
        selection = view(owners, [3, 1], 2:3)
        copied_owners, copied_selection = specialized_deepcopy((owners, selection))
        @test copied_selection[1, 1] === copied_owners[3, 2] !== owners[3, 2]
        @test parent(specialized_deepcopy(selection)) !== owners

        copied_owners, copied_permuted = specialized_deepcopy((owners, PermutedDimsArray(owners, (2, 1))))
        @test copied_permuted[3, 1] === copied_owners[1, 3]

        copied_columns, copied_owners = specialized_deepcopy((eachcol(owners), owners))
        @test copied_columns[2][1] === copied_owners[1, 2]

        shifted = OffsetArray(view(owners, 1:2, 1:2), -1, 5)
        copied = specialized_deepcopy(shifted)
        @test axes(copied) == axes(shifted)
        @test copied[0, 6] !== owners[1, 1] && copied[0, 6].id == owners[1, 1].id

        values = collect(1.0:6.0)
        copied_values, copied_reshaped = specialized_deepcopy((values, reshape(view(values, 2:5), 2, 2)))
        copied_reshaped[1, 1] = 20.0
        @test copied_values[2] == 20.0 && values[2] == 2.0

        matrix = [1.0 2.0; 3.0 4.0]
        copied_matrix, copied_transpose, copied_symmetric = specialized_deepcopy((matrix, transpose(matrix), Symmetric(matrix, :L)))
        copied_matrix[2, 1] = 30.0
        @test copied_transpose[1, 2] == 30.0 && copied_symmetric[1, 2] == 30.0 && matrix[2, 1] == 3.0
        @test specialized_deepcopy(Diagonal([1.0, 2.0])) == Diagonal([1.0, 2.0])

        sparse_matrix = sparse([1, 2], [1, 2], [1.0, 2.0])
        copied = specialized_deepcopy(sparse_matrix)
        nonzeros(copied)[1] = 10.0
        @test copied[1, 1] == 10.0 && sparse_matrix[1, 1] == 1.0

        bits = falses(2, 3)
        bits[1, 2] = true
        copied_bits, copied_row = specialized_deepcopy((bits, view(bits, 1, :)))
        copied_row[2] = false
        @test !copied_bits[1, 2] && bits[1, 2]

        storage = Storage([1.0, 2.0, 3.0])
        copied_left, copied_right = specialized_deepcopy((Window(storage, 1:2), Window(storage, 2:3)))
        copied_left[2] = 20.0
        @test copied_right[1] == 20.0 && storage.data[2] == 2.0

        shared = [1.0, 2.0, 3.0]
        copied = specialized_deepcopy(Any[shared, view(shared, 1:2), shared])
        @test copied[1] === copied[3] !== shared && parent(copied[2]) === copied[1]

        for lazy in (1:5, range(0.0, 1.0; length = 11), Base.OneTo(3), LinRange(0, 1, 5), CartesianIndices((2, 3)))
            @test specialized_deepcopy(lazy) === lazy
        end
        cube = [Owner(i) for i in 1:2, j in 1:2, k in 1:2]
        copied = specialized_deepcopy(cube)
        @test size(copied) == (2, 2, 2) && copied[2, 2, 2].id == 2.0 && copied[2, 2, 2] !== cube[2, 2, 2]
    end

    @testset "abstract dicts" begin
        owner = Owner(1.0)
        nested = Base.ImmutableDict(Base.ImmutableDict(owner => [1.0]), Owner(2.0) => [2.0])
        copied = specialized_deepcopy(nested)
        copied_owner = only(k for k in keys(copied) if k.id == 1.0)
        @test copied[copied_owner] == [1.0] && copied_owner !== owner && !haskey(copied, owner)

        keywords = pairs((weights = [1.0, 2.0], owner = owner))
        copied = specialized_deepcopy(keywords)
        @test copied[:weights] == [1.0, 2.0] && copied[:weights] !== keywords[:weights] && copied[:owner] !== owner

        @test specialized_deepcopy(ENV) === ENV

        key = [1, 2]
        copied = specialized_deepcopy(Dict(key => "a"))
        @test copied[key] == "a" && only(keys(copied)) !== key

        cyclic = Dict{String, Any}()
        cyclic["self"] = cyclic
        copied = specialized_deepcopy(cyclic)
        @test copied["self"] === copied !== cyclic

        weights = [1.0]
        first_table, second_table = specialized_deepcopy((Dict("a" => weights), Dict("b" => weights)))
        @test first_table["a"] === second_table["b"] !== weights

        @test specialized_deepcopy(BitSet([1, 5, 100])) == BitSet([1, 5, 100])

        # Stored objectids go stale unless the table is rebuilt.
        table = IdentityTable{Owner, Int}()
        table[owner] = 1
        copied = specialized_deepcopy(table)
        @test !haskey(copied, only(keys(copied)))
        rebuilt = RebuiltIdentityTable{Owner, Int}()
        rebuilt[owner] = 1
        copied, again = specialized_deepcopy((rebuilt, rebuilt))
        @test copied === again && copied[only(keys(copied))] == 1 && !haskey(copied, owner)

        @test_throws PointerCopyError(WeakRef, nothing) specialized_deepcopy(WeakKeyDict(owner => 1))
    end

    # Compare generic copying with overrides that preserve each resource's invariants.
    @testset "situations that need a method" begin
        # Base.deepcopy keeps the address of C memory, and a method duplicates the memory.
        test_externally_allocated_memory()

        # The strdup copy must survive finalizing the original string owner.
        text = CText("parameters")
        copied, again = specialized_deepcopy((text, text))
        @test copied === again && copied.text != text.text
        finalize(text)
        @test unsafe_string(copied.text) == "parameters"

        # Base leaves an interior pointer aimed at the original array.
        bytes = UInt8[1, 2, 3]
        @test deepcopy(ByteView(bytes, pointer(bytes, 2))).cursor == pointer(bytes, 2)
        @test_throws PointerCopyError(ByteView, :cursor) specialized_deepcopy(ByteView(bytes, pointer(bytes, 2)))
        # The override redirects the pointer into the copied array.
        copied = specialized_deepcopy(OwnedByteView(bytes, pointer(bytes, 2)))
        GC.gc()
        bytes[2] = 0x09
        GC.@preserve copied begin
            @test copied.cursor == pointer(copied.bytes, 2) && unsafe_load(copied.cursor) == 0x02
        end

        # The owner's override explicitly shares pointed-to C objects.
        address = Ptr{Cvoid}(UInt(0x1000))
        @test_throws PointerCopyError(Ptr{Cvoid}, nothing) specialized_deepcopy([address])
        table = PointerTable([address])
        copied = specialized_deepcopy(table)
        @test copied.addresses == [address] && copied.addresses !== table.addresses

        # The extension points the copy of a closure's wrapper at the copied closure.
        weights = [1.0]
        wrapper = FunctionWrapper{Float64, Tuple{Float64}}(x -> weights[1] * x)
        copied, again = specialized_deepcopy((wrapper, wrapper))
        weights[1] = 2.0
        @test copied === again && copied(1.0) == 1.0 && wrapper(1.0) == 2.0
        # A stateful callable, as ModelingToolkit wraps `f(::Real)`, and Base.deepcopy's copy of it.
        test_wrapped_mutable_callable()
        # A copied wrapper whose callable holds another wrapper calls the copied inner state.
        for Twice in (InlineTwice, MutableTwice)
            inner = FunctionWrapper{Float64, Tuple{Float64}}(let w = [3.0]; x -> w[1] * x end)
            outer = FunctionWrapper{Float64, Tuple{Float64}}(Twice(inner))
            copied_outer, copied_inner = specialized_deepcopy((outer, inner))
            @test copied_outer.obj[].inner === copied_inner !== inner
            copied_inner.obj[].w[1] = 5.0
            @test copied_outer(1.0) == 10.0 && outer(1.0) == 6.0 && copied_inner(1.0) == 5.0
        end
        # Memoize the wrapper before following its callable's back-reference.
        law = SelfAwareLaw(nothing)
        wrapper = FunctionWrapper{Float64, Tuple{Float64}}(law)
        law.wrapper = wrapper
        for value in ((wrapper, law), (law, wrapper))
            copied = specialized_deepcopy(value)
            copied_wrapper, copied_law = value[1] isa FunctionWrapper ? copied : reverse(copied)
            @test copied_law.wrapper === copied_wrapper !== wrapper && copied_wrapper.obj[] === copied_law !== law
            @test copied_wrapper(2.0) == 2.0
        end
        # Every traversal order must preserve wrapper, Ref and callable aliases.
        ref = wrapper.obj
        for value in ((wrapper, ref, law), (wrapper, law, ref), (ref, wrapper, law), (ref, law, wrapper), (law, wrapper, ref), (law, ref, wrapper))
            copied = specialized_deepcopy(value)
            copied_wrapper = copied[findfirst(v -> v === wrapper, value)]
            copied_ref = copied[findfirst(v -> v === ref, value)]
            copied_law = copied[findfirst(v -> v === law, value)]
            @test copied_wrapper.obj === copied_ref && copied_ref[] === copied_law && copied_law.wrapper === copied_wrapper
            @test copied_wrapper !== wrapper && copied_ref !== ref && copied_law !== law && copied_wrapper(2.0) == 2.0
            @test copied_wrapper.objptr == convert(Ptr{Cvoid}, Base.unsafe_convert(Ref{SelfAwareLaw}, copied_ref))
        end
        @test specialized_deepcopy(FunctionWrapper{Int, Tuple{Int}}(PlainId(3)))(2) == 6
        @test_throws TypeError specialized_deepcopy(FunctionWrapper{Int, Tuple{Int}}(WrongType(1)))
        # The conversion root need not be a Ref.
        own = FunctionWrapper{Int, Tuple{Int}}(OwnRootLaw(3))
        copied = specialized_deepcopy(own)
        @test copied.obj isa OwnRootLaw && copied.obj !== own.obj && copied(2) == 6
        copied.obj.gain = 5
        @test copied(2) == 10 && own(2) == 6
        # Custom roots defer pointer conversion until the first call, after the cycle is complete.
        rooted = FunctionWrapper{Int, Tuple{Int}}(RootedLaw(3))
        rooted.obj.wrapper = rooted
        for value in ((rooted, rooted.obj), (rooted.obj, rooted))
            copied = specialized_deepcopy(value)
            copied_wrapper, copied_root = value[1] isa FunctionWrapper ? copied : reverse(copied)
            @test copied_wrapper.obj === copied_root && copied_root.wrapper === copied_wrapper && copied_root !== rooted.obj
            @test copied_wrapper.ptr == C_NULL
            @test copied_wrapper(2) == 6
            @test copied_wrapper.objptr == convert(Ptr{Cvoid}, Base.unsafe_convert(Ref{RootedLaw}, copied_root))
        end
        # Array storage can move during copying; defer conversion for both callable layouts and orders.
        for law in (ArrayRootLaw(3, nothing), ImmutableArrayRootLaw(3, Ref{Any}()))
            wrapper = FunctionWrapper{Int, Tuple{Int}}(law)
            law isa ArrayRootLaw ? (law.back = wrapper) : (law.back[] = wrapper)
            for value in ((wrapper, wrapper.obj), (wrapper.obj, wrapper))
                copied = specialized_deepcopy(value)
                copied_wrapper, copied_root = value[1] isa FunctionWrapper ? copied : reverse(copied)
                @test copied_wrapper.obj === copied_root && copied_root.x !== wrapper.obj.x && copied_root.x[1] !== law
                @test copied_wrapper.ptr == C_NULL && copied_wrapper.objptr == C_NULL
                @test copied_wrapper(2) == 6
                @test copied_wrapper.objptr == convert(Ptr{Cvoid}, Base.unsafe_convert(Ref{typeof(law)}, copied_root))
            end
        end
        # Defer conversion through an enclosing, still-incomplete storage object.
        stored = FunctionWrapper{Int, Tuple{Int}}(StoredLaw(3))
        stored.obj.storage.wrapper = stored
        copied_storage, copied_wrapper = specialized_deepcopy((stored.obj.storage, stored))
        @test copied_wrapper.obj.storage === copied_storage && copied_storage.wrapper === copied_wrapper
        @test copied_wrapper.ptr == C_NULL && copied_wrapper(2) == 6
        @test copied_wrapper.objptr == convert(Ptr{Cvoid}, Base.unsafe_convert(Ref{StoredLaw}, copied_wrapper.obj))
        # Plain roots bypass overrides.
        static = FunctionWrapper{Int, Tuple{Int}}(StaticLaw())
        copied = specialized_deepcopy(static)
        @test copied.obj === static.obj && copied(3) == 6
        # A cycle reaches the Ref of a boxed immutable callable first. Filling the unfinished Ref
        # copy would point the wrapper at a box that the Ref copy then replaces.
        gain = LawGain(nothing, 3)
        boxed = FunctionWrapper{Int, Tuple{Int}}(BoxedLaw(gain, [1]))
        gain.wrapper = boxed
        @test !Base.allocatedinline(BoxedLaw)
        for value in ((boxed.obj, boxed), (boxed.obj, gain, boxed))
            copied = specialized_deepcopy(value)
            copied_ref, copied_wrapper = first(copied), last(copied)
            @test copied_wrapper.obj === copied_ref && copied_ref[].state.wrapper === copied_wrapper
            GC.gc()
            gain.gain = 5
            @test copied_wrapper(2) == 6
            @test copied_wrapper.objptr == convert(Ptr{Cvoid}, Base.unsafe_convert(Ref{BoxedLaw}, copied_ref))
            gain.gain = 3
        end
        # Custom Ref conversion must wait: copying the state first reaches the wrapper while
        # the copied state still holds the original callable.
        stateful = FunctionWrapper{Int, Tuple{Int}}(Fuzz.StateLaw(3))
        state = stateful.obj[]
        state.wrapper = stateful
        for value in ((stateful, state), (state, stateful))
            copied = specialized_deepcopy(value)
            copied_wrapper, copied_state = value[1] isa FunctionWrapper ? copied : reverse(copied)
            @test copied_wrapper.ptr == C_NULL && copied_wrapper.obj[] === copied_state && copied_state.wrapper === copied_wrapper
            GC.gc()
            @test copied_wrapper(2) == 6
            @test copied_wrapper.objptr == convert(Ptr{Cvoid}, Base.unsafe_convert(Ref{Fuzz.StateLaw}, copied_wrapper.obj))
            @test copied_wrapper.objptr != stateful.objptr
        end
        # Defer custom pointer conversion for Base.cconvert's Ref too.
        own_conversion = specialized_deepcopy(FunctionWrapper{Int, Tuple{Int}}(PtrLaw(3)))
        @test own_conversion.ptr == C_NULL && own_conversion(2) == 6

        # Rebuild indexes derived from copied object identities.
        owners = [Owner(1.0), Owner(2.0)]
        stale = specialized_deepcopy(StaleOwnerIndex(owners))
        @test !haskey(stale.slots, objectid(stale.owners[2]))
        copied = specialized_deepcopy(OwnerIndex(owners))
        @test copied.owners[2] !== owners[2] && copied.slots[objectid(copied.owners[2])] == 2

        # Only the override runs the numbering constructor again.
        ticket = Ticket()
        @test specialized_deepcopy(ticket).number == ticket.number
        numbered = NumberedTicket()
        copied = specialized_deepcopy(numbered)
        @test copied.number != numbered.number && copied.number == TICKET_COUNT[]

        # Share the stream explicitly instead of copying its internal C handle.
        mktemp() do path, io
            @test deepcopy(io).handle == io.handle
            @test_throws PointerCopyError(IOStream, :handle) specialized_deepcopy(LoggerWithoutMethod(io, 1))
            copied = specialized_deepcopy(Record(Logger(io, 1)))
            print(copied.value.io, "shared")
            flush(io)
            @test copied.value.io === io && read(path, String) == "shared"
        end

        # The method clears the cache instead of copying weak keys.
        test_result_cache()

        # Registry's assertion removes dispatch; the trim suite also tests the unhandled abstract field.
        @test specialized_deepcopy(Registry(Owner(3.0))).owner.id == 3.0
    end

    @testset "Julia internals" begin
        # `n_uninitialized` counts the last fields that a constructor can leave undefined.
        @test (Owner.name.n_uninitialized, Lazy.name.n_uninitialized, Prefix.name.n_uninitialized) == (0, 1, 2)
        @test SpecializedDeepcopy._may_be_undefined(Lazy, 2) && !SpecializedDeepcopy._may_be_undefined(Lazy, 1)
        # Inline bits Unions are always defined; inline references and atomic Unions can be undefined.
        @test !SpecializedDeepcopy._may_be_undefined(UnionTail, 2)
        @test SpecializedDeepcopy._may_be_undefined(LateTuple, 1)
        @test SpecializedDeepcopy._may_be_undefined(AtomicUnionTail, 2)
        @test specialized_deepcopy(UnionTail(1)).id == 1
        copied = specialized_deepcopy(AtomicUnionTail(1))
        @test copied.id == 1 && !isdefined(copied, :tag, :sequentially_consistent)
        tail = AtomicUnionTail(1)
        @atomic tail.tag = 2
        @test (@atomic specialized_deepcopy(tail).tag) == 2

        # Skip `isdefined` on an uninhabited field: after another field check, it breaks Julia 1.13 inference.
        for T in (RefThenNever, MutableRefThenNever)
            @test only(Base.return_types(specialized_deepcopy, (T,))) === T
            copied = specialized_deepcopy(T(1, [2.0]))
            @test copied.id == 1 && copied.weights == [2.0] && !isdefined(copied, :value)
            @test !isdefined(specialized_deepcopy(T(1)), :weights)
        end

        # An unset field without references can hold isbits Union selectors that are not valid. The
        # copy uses the bits of such a value and does not read its fields.
        @test SpecializedDeepcopy._copies_bits(AtomicPair, 2) && SpecializedDeepcopy._copies_bits(UnsetPair, 2)
        @test !SpecializedDeepcopy._copies_bits(UnsetPair, 1) && !SpecializedDeepcopy._copies_bits(TaggedHolder, 1)
        @test all(p -> specialized_deepcopy(p).id == p.id, [AtomicPair(i) for i in 1:1000])
        @test (@atomic specialized_deepcopy(AtomicPair(1, (a = 2, b = Int8(3)))).pair) === UnionPair((2, Int8(3)))
        @test [p.id for p in specialized_deepcopy([UnsetPair(i) for i in 1:1000])] == 1:1000

        # Under --compile=min, `new` requires the fields initialized by every constructor.
        # Supply those fields when allocating the copy.
        code = """
            using SpecializedDeepcopy
            mutable struct Node
                label::String
                next::Union{Nothing, Node}
            end
            node = Node("loop", nothing)
            node.next = node
            copied = specialized_deepcopy((node, [node], Dict(1 => node)))
            print(copied[1].next === copied[1] === copied[2][1] === copied[3][1] !== node)
            """
        @test run_isolated(code, `--compile=min`) == "true"
        # Without compiled modules, Julia 1.12 and later resolve the bindings of a generated function
        # in source order; the @cfunction of the table check needs its function defined first.
        @test run_isolated("using SpecializedDeepcopy; print(specialized_deepcopy(Dict([1] => 2))[[1]])", `--compiled-modules=no`) == "2"

        # The recursion relation is set on the loaded, precompiled package.
        @test all(m -> isdefined(m, :recursion_relation) && m.recursion_relation === SpecializedDeepcopy._copy_recursion_ends,
            methods(SpecializedDeepcopy._copy_value))
        @test SpecializedDeepcopy._type_depth(Union{Int, Vector{Vector{Int}}}) == 3
        @test SpecializedDeepcopy._type_depth(Vector) == 1

        # A zero-field mutable object still needs a distinct, memoized copy.
        token = Token()
        copied, again = specialized_deepcopy((token, token))
        @test copied === again !== token
        # Builtins with hidden contents need explicit methods rather than field copying.
        @test all(SpecializedDeepcopy._has_hidden_contents, (Core.SimpleVector, String, Symbol, Module))
        @test SpecializedDeepcopy._has_hidden_contents(Vector{Int}) == !isdefined(Base, :Memory)
        @test !SpecializedDeepcopy._has_hidden_contents(Token) && !SpecializedDeepcopy._has_hidden_contents(Owner)
        @test_throws ArgumentError SpecializedDeepcopy._copy_by_fields(Core.svec(1), IdDict{Any, Any}())
        vector = Core.svec(1, [2.0])
        copied, again = specialized_deepcopy((vector, vector))
        @test copied === again !== vector && copied[1] == 1 && copied[2] == [2.0] && copied[2] !== vector[2]

        # Const fields go through `jl_set_nth_field`.
        fixed = Fixed(1, 2, [3.0])
        copied = specialized_deepcopy(fixed)
        @test (copied.id, copied.tag, copied.weights) == (1, 2, [3.0]) && copied.weights !== fixed.weights
        @test specialized_deepcopy(Fixed(1, nothing, Float64[])).tag === nothing
        @test !isdefined(specialized_deepcopy(ConstLater()), :weights)

        # Check undefined and initialized atomic reference fields.
        @test !isdefined(specialized_deepcopy(AtomicLater()), :weights)
        later = AtomicLater()
        @atomic later.weights = [1.0]
        copied = specialized_deepcopy(later)
        @test (@atomic copied.weights) == [1.0] && (@atomic copied.weights) !== (@atomic later.weights)

        # Expand Unions and tuple variants only within MAX_SPLIT_TYPES.
        concrete_types = SpecializedDeepcopy._concrete_types
        @test Set(concrete_types(Union{Nothing, Int})) == Set([Nothing, Int])
        @test Set(concrete_types(Tuple{Float64, Union{Missing, Float64}})) == Set([Tuple{Float64, Missing}, Tuple{Float64, Float64}])
        @test length(concrete_types(Union{Nothing, Tuple{Int, Union{Missing, Int}}})) == 3
        @test length(concrete_types(Union{Nothing, Int, Float64, String, Owner})) == 5
        @test concrete_types(AbstractVector{Float64}) === nothing && concrete_types(Tuple{Vararg{Int}}) === nothing
        @test length(concrete_types(Tuple{ntuple(_ -> Union{Missing, Int}, 4)...})) == 16
        @test concrete_types(Tuple{ntuple(_ -> Union{Missing, Int}, 5)...}) === nothing
        # 2^63 variants overflow a signed 64-bit count.
        @test concrete_types(Tuple{ntuple(_ -> Union{Missing, Int}, 63)...}) === nothing
        wide = Wide(ntuple(_ -> missing, 63))
        @test specialized_deepcopy(wide).values === wide.values

        # Split concrete Union alternatives; leave abstract alternatives to dispatch.
        @test specialized_deepcopy(ThreeWay(Owner(1.0))).value.id == 1.0
        @test specialized_deepcopy(ThreeWay(2)).value == 2
        @test specialized_deepcopy(ThreeWay(nothing)).value === nothing
        @test specialized_deepcopy(FiveWay(Owner(1.0))).value.id == 1.0
        @test specialized_deepcopy(FiveWay("text")).value == "text"
        @test specialized_deepcopy(OptionalVector(view([1.0, 2.0], 1:1))).data == [1.0]

        @test_throws PointerCopyError(Cwstring, nothing) specialized_deepcopy(reinterpret(Cwstring, C_NULL))
        llvm_pointer = reinterpret(Core.LLVMPtr{Float32, 1}, C_NULL)
        @test_throws PointerCopyError(typeof(llvm_pointer), nothing) specialized_deepcopy(llvm_pointer)

        # The Base types that `Base.deepcopy_internal` shares or rebuilds.
        reference = GlobalRef(Base, :sin)
        @test specialized_deepcopy(reference) === reference
        method = first(methods(sin))
        @test specialized_deepcopy(method) === method
        instance = first(Base.method_instances(sin, (Float64,), Base.get_world_counter()))
        @test specialized_deepcopy(instance) === instance
        spin = Threads.SpinLock()
        lock(spin)
        @test !islocked(specialized_deepcopy(spin))
        unlock(spin)
        event = Threads.Event()
        notify(event)
        copied = specialized_deepcopy(event)
        wait(copied)
        @test copied !== event && copied.notify.lock !== event.notify.lock
        semaphore = Base.Semaphore(2)
        Base.acquire(semaphore)
        copied = specialized_deepcopy(semaphore)
        Base.acquire(copied)
        @test copied.curr_cnt == 2 && semaphore.curr_cnt == 1
        # A channel's conditions share one lock, and so do the conditions of its copy.
        channel = Channel{Int}(4)
        put!(channel, 1)
        copied = specialized_deepcopy(channel)
        put!(copied, 2)
        @test take!(copied) == 1 && take!(copied) == 2 && take!(channel) == 1 && !isready(channel)
        @test copied.cond_take.lock === copied.cond_put.lock !== channel.cond_take.lock

        # Arrays with spare capacity, with an offset in their storage, and without elements.
        grown = Owner[]
        sizehint!(grown, 10)
        push!(grown, Owner(1.0))
        @test only(specialized_deepcopy(grown)).id == 1.0
        # The copy also reads the unused storage. For these element types, it holds Union selectors.
        grown_union = Union{Missing, Float64}[1.0, missing]
        grown_inline = [Some{Union{Int8, UInt8}}(Int8(1))]
        sizehint!(grown_union, 64)
        sizehint!(grown_inline, 64)
        @test isequal(specialized_deepcopy(grown_union), grown_union) && specialized_deepcopy(grown_inline) == grown_inline
        shortened = [Owner(1.0), Owner(2.0), Owner(3.0)]
        deleteat!(shortened, 1)
        copied = specialized_deepcopy(shortened)
        push!(copied, Owner(4.0))
        @test length(copied) == 3 && copied[1].id == 2.0 && length(shortened) == 2
        emptied = [Owner(1.0)]
        empty!(emptied)
        @test isempty(specialized_deepcopy(emptied)) && isempty(specialized_deepcopy(Owner[]))

        # Memory and MemoryRef exist from Julia 1.11.
        @static if isdefined(Base, :Memory)
            memory = Memory{Owner}(undef, 3)
            memory[1] = Owner(1.0)
            memory[3] = memory[1]
            copied_memory, copied_reference = specialized_deepcopy((memory, memoryref(memory, 3)))
            @test !isassigned(copied_memory, 2) && copied_memory[1] === copied_memory[3] !== memory[1]
            @test copied_reference[] === copied_memory[3]
            @test specialized_deepcopy(Memory{Float64}(undef, 0)) isa Memory{Float64}
        end
    end

    @testset "recursion relation" begin
        signature(T) = Tuple{typeof(SpecializedDeepcopy._copy_value), T, IdDict{Any, Any}}
        ends(T) = SpecializedDeepcopy._copy_recursion_ends(nothing, nothing, signature(T), signature(Owner))
        # Record nested k times around an Owner has type depth k + 1.
        nested_type(k) = typeof(nested_records(Owner(0.0), Val(k)))
        limit = SpecializedDeepcopy.MAX_INFERRED_TYPE_DEPTH
        @test ends(nested_type(limit - 1))
        @test !ends(nested_type(limit))

        # Log actual inference calls using uncached Layer types: no refusal at the bound,
        # one refusal just beyond it.
        refused = Any[]
        logged(method, callee, sig, earlier) = begin
            ok = SpecializedDeepcopy._copy_recursion_ends(method, callee, sig, earlier)
            ok || push!(refused, sig)
            ok
        end
        method = only(methods(SpecializedDeepcopy._copy_value))
        method.recursion_relation = logged
        try
            for (k, refusals) in ((limit, 0), (limit + 1, 1))
                empty!(refused)
                value = (nested_layers(Owner(0.0), Val(k)),)
                Base.code_typed(SpecializedDeepcopy._copy_value, (typeof(value), IdDict{Any, Any}))
                @test length(refused) == refusals
                @test all(sig -> SpecializedDeepcopy._type_depth(Base.unwrap_unionall(sig).parameters[2]) == limit + 1, refused)
            end
        finally
            method.recursion_relation = SpecializedDeepcopy._copy_recursion_ends
        end
    end

    include("deepcopy_internal_methods.jl")

    # Each item of "Shared with Base.deepcopy" in the README, for both implementations. The item about
    # initialization rules has no test: breaking those rules is undefined behavior.
    @testset "limits shared with Base.deepcopy" begin
        copiers = (specialized_deepcopy, deepcopy)
        # Both implementations are recursive: 1000 nodes copy, and a million overflow the stack of a fresh process.
        chain(n) = foldl((next, _) -> Node("node", next), 1:n; init = nothing)
        length_of(node) = (n = 0; while node !== nothing; n += 1; node = node.next; end; n)
        @test all(copier -> length_of(copier(chain(1000))) == 1000, copiers)
        # Each overflow runs in a process of its own. After a stack overflow, the state of the process can
        # be corrupt, and a second overflow can crash it on Windows (see JuliaLang/julia#46870). The
        # handler is in a function, which Julia compiles: on Windows from Julia 1.13, an overflow that
        # unwinds to a handler in interpreted top-level code crashes the process.
        for copier in (:specialized_deepcopy, :deepcopy)
            code = """
                using SpecializedDeepcopy
                include($(repr(joinpath(@__DIR__, "types.jl"))))
                function overflow(copy, value)
                    try
                        copy(value)
                        return "copied"
                    catch e
                        return string(nameof(typeof(e)))
                    end
                end
                long = foldl((next, _) -> Node("node", next), 1:1_000_000; init = nothing)
                print(overflow($copier, long))
                flush(stdout)
                """
            @test run_outcome(code) == (true, "StackOverflowError")
        end
        # A method that returns the wrong type, and a module.
        for copier in copiers
            @test_throws TypeError copier(Misfit(1))
            @test_throws TypeError copier(MisfitHolder(Misfit(1)))
            @test_throws TypeError copier([Misfit(1)])
            @test_throws ErrorException copier(Base)
            @test_throws ErrorException copier((1, Base))
        end
        # A reshape and the sdata array use the segment of the original SharedArray, but not of its copy.
        array = SharedArray{Float64}(2; pids = [1])
        array .= 0.0
        for copier in copiers
            copied, backing, shaped = copier((array, sdata(array), reshape(array, 1, 2)))
            copied[1] = 9.0
            @test backing[1] == 0.0 && shaped[1] == 0.0
        end
        # From Julia 1.11, neither implementation copies a GenericMemory other than Memory. This
        # package raises a PointerCopyError that names the field, and Base.deepcopy a TypeError.
        @static if isdefined(Base, :AtomicMemory)
            atomic = Base.AtomicMemory{Int}(undef, 2)
            @test_throws PointerCopyError(Base.AtomicMemory{Int}, :ptr) specialized_deepcopy(atomic)
            @test_throws TypeError deepcopy(atomic)
            message = sprint(showerror, PointerCopyError(Base.AtomicMemory{Int}, :ptr))
            @test occursin("`ptr`", message) && occursin("Add a method", message)
        end
    end

    # Each item of "Different from Base.deepcopy" in the README, for both implementations.
    @testset "limits that differ from Base.deepcopy" begin
        # A key whose hash reads a field that the copy has not finished: Base.deepcopy crashes the
        # process, and this package completes, with a method that fixes the key's table.
        test_cyclic_key()

        # From Julia 1.11, Base.deepcopy keeps the stale objectids of the original members in the
        # index of the copied IdSet.
        members = [Owner(Float64(i)) for i in 1:5]
        set = Base.IdSet{Owner}(members)
        storage = getfield(set, 1)  # the IdDict before Julia 1.11, the member list from 1.11
        copied, copied_storage = specialized_deepcopy((set, storage))
        @test length(copied) == 5 && all(m -> m in copied, collect(copied)) && !any(m -> m in copied, members)
        @test (getfield(copied, 1) === copied_storage) == (VERSION < v"1.11")
        base_copied, base_storage = deepcopy((set, storage))
        @test getfield(base_copied, 1) === base_storage
        @test all(m -> m in base_copied, collect(base_copied)) == (VERSION < v"1.11")
    end

    # Check the oracle's rejection cases, then run fixed seeds with copier and family diagnostics.
    @testset "fuzz" begin
        vector = [1.0]
        @test Fuzz.check(vector, vector) !== nothing
        @test Fuzz.check((vector, vector), ([1.0], [1.0])) !== nothing
        @test Fuzz.check((vector, [1.0]), let other = [1.0]; (other, other) end) !== nothing
        @test Fuzz.check(vector, [1.0]) === nothing
        set = Set([1])
        @test Fuzz.check(set, set) !== nothing
        # With `trim`, the generator keeps the 16 tuple types of `wide_tuple(4)` out of a Union, which
        # would exceed the split limit and fail the trim build of a fixed seed on Julia 1.12 and later.
        wide_in_union(ex) = ex isa Expr && ((ex.head === :curly && ex.args[1] === :Union &&
            any(arg -> arg === Fuzz.wide_tuple(4), ex.args)) || any(wide_in_union, ex.args))
        draws(trim) = (Fuzz.random_type(Xoshiro(seed), DataType[], :Self, 2; trim, methods = true) for seed in 1:5000)
        @test any(wide_in_union, draws(false)) && !any(wide_in_union, draws(true))
        # The generator sets every field, Ref content and array element whose unset state `isdefined`
        # cannot see, because that state holds arbitrary bytes. An unset array element shows the state
        # for a type. An uninitialized mutable object, as Base.deepcopy allocates one, shows it for a
        # field, which can be atomic. An immutable object with unset required fields is invalid.
        unseen = Any[]
        fuzz_types = [FUZZ_FAMILY; FUZZ_METHOD_FAMILY; FUZZ_TRIM_FAMILY]
        listed = Any[Int, Nothing, Union{Nothing, Int}, Some{Union{Int8, UInt8}}, Tuple{Float64, Union{Missing, Float64}},
            Some{Vector{Int}}, Union{Nothing, Vector{Int}}, Owner]
        for F in unique([listed; [F for T in fuzz_types for F in fieldtypes(T)]])
            isassigned(Vector{F}(undef, 1), 1) && !Fuzz.inline_type(F) && push!(unseen, F)
            isdefined(Base.RefValue{F}(), :x) && !Fuzz.inline_type(F) && push!(unseen, Base.RefValue{F})
        end
        for T in filter(ismutabletype, fuzz_types), j in 1:fieldcount(T)
            unset = ccall(:jl_new_struct_uninit, Any, (Any,), T)
            isdefined(unset, j, Fuzz.order(T, j)...) && !Fuzz.inline(T, j) && push!(unseen, (T, j))
        end
        @test isempty(unseen) && Fuzz.inline_type(Some{Union{Int8, UInt8}}) && !Fuzz.inline_type(Some{Vector{Int}})
        # The oracle expects a bits copy exactly where the package makes one.
        @test all(T -> all(j -> Fuzz.bits_field(T, j) == SpecializedDeepcopy._copies_bits(T, j), 1:fieldcount(T)), fuzz_types)
        # A wrapper awaiting its first call must itself be copyable.
        rooted = FunctionWrapper{Int, Tuple{Int}}(RootedLaw(3))
        rooted.obj.wrapper = rooted
        lazy = specialized_deepcopy((rooted.obj, rooted))[2]
        @test lazy.ptr == C_NULL && Fuzz.check(lazy, specialized_deepcopy(lazy)) === nothing
        # Undefined fields must not hide later initialized fields from the oracle.
        gapped, other, sharing = Gapped(), Gapped(), Gapped()
        gapped.second = [1]
        other.second = [9]
        sharing.second = gapped.second
        @test Fuzz.check(gapped, other) !== nothing && Fuzz.check(gapped, sharing) !== nothing
        @test Fuzz.check(gapped, specialized_deepcopy(gapped)) === nothing
        @static if isdefined(Base, :Memory)
            # Reject original-storage sharing and lost aliases; accept one independent shared Memory.
            flat = [1, 2]
            @test Fuzz.check(flat, reshape(reshape(flat, 1, 2), 2)) !== nothing
            @test Fuzz.check((flat, reshape(flat, 1, 2)), (copy(flat), copy(reshape(flat, 1, 2)))) !== nothing
            @test Fuzz.check((flat, reshape(flat, 1, 2)), let other = copy(flat); (other, reshape(other, 1, 2)) end) === nothing
            # Arrays keep their offsets into a shared Memory.
            segment(memory, i) = Base.wrap(Array, memoryref(memory, i), (1,))
            memory, other_memory = fill!(Memory{Int}(undef, 2), 1), fill!(Memory{Int}(undef, 2), 1)
            segments = (segment(memory, 1), segment(memory, 2))
            @test Fuzz.check(segments, (segment(other_memory, 1), segment(other_memory, 1))) !== nothing
            @test Fuzz.check(segments, (segment(other_memory, 1), segment(other_memory, 2))) === nothing
            # Check the whole backing Memory, including elements outside the array, in both root orders.
            hidden = Memory{Vector{Int}}(undef, 2)
            hidden[1], hidden[2] = [1], [2]
            visible = Base.wrap(Array, memoryref(hidden, 1), (1,))
            wrong = Memory{Vector{Int}}(undef, 2)
            wrong[1], wrong[2] = [1], [999]
            wrong_visible = Base.wrap(Array, memoryref(wrong, 1), (1,))
            @test Fuzz.check((visible, hidden), (wrong_visible, wrong)) !== nothing
            @test Fuzz.check((hidden, visible), (wrong, wrong_visible)) !== nothing
            wrong[2] = hidden[2]
            @test Fuzz.check((visible, hidden), (wrong_visible, wrong)) !== nothing
            @test Fuzz.check((visible,), (wrong_visible,)) !== nothing
            @test Fuzz.check((visible, hidden), specialized_deepcopy((visible, hidden))) === nothing
        end
        # Tables with fixture keys: the checker pairs entries by key ID and compares keys like values.
        # A key in a stale slot: the only entry of a copied table moves to another slot, so a lookup
        # from the key's hash reaches an empty slot.
        function move_entry!(d::Dict)
            i = findfirst(j -> Base.isslotfilled(d, j), eachindex(d.slots))
            j = i == lastindex(d.slots) ? firstindex(d.slots) : i + 1
            d.keys[j], d.vals[j], d.slots[j], d.slots[i] = d.keys[i], d.vals[i], d.slots[i], 0x00
            d.idxfloor = firstindex(d.slots)
            return d
        end
        table = Dict(Fuzz.IdentityKey(1) => [1])
        @test Fuzz.check(table, specialized_deepcopy(table)) === nothing
        @test occursin("does not find its entry", Fuzz.check(table, move_entry!(specialized_deepcopy(table))))
        # A key and a value that share an owner, and a copy where they do not.
        owner = [1]
        shared = Dict(Fuzz.ContentKey(1, owner) => owner)
        @test Fuzz.check(shared, specialized_deepcopy(shared)) === nothing
        @test occursin("not shared by the copy", Fuzz.check(shared, Dict(Fuzz.ContentKey(1, [1]) => [1])))
        # Distinct keys that become one: two copied keys with one ID, or one entry for two.
        ids = IdDict(Fuzz.IdentityKey(1) => 1, Fuzz.IdentityKey(2) => 2)
        @test occursin("two keys of the copy", Fuzz.check(ids, IdDict(Fuzz.IdentityKey(1) => 1, Fuzz.IdentityKey(1) => 2)))
        contents = Dict(Fuzz.ContentKey(1, [1]) => 1, Fuzz.ContentKey(2, [2]) => 2)
        @test occursin("length 2 became 1", Fuzz.check(contents, Dict(Fuzz.ContentKey(1, [1]) => 1)))
        # A copy method that changes a tuple key, and a copy without the change.
        tagged = Dict((Fuzz.Tag(1), "a") => 1)
        @test Fuzz.check(tagged, specialized_deepcopy(tagged)) === nothing
        @test occursin("did not run", Fuzz.check(tagged, Dict((Fuzz.Tag(1), "a") => 1)))
        # Two original keys with one ID make the pairing ambiguous.
        @test_throws ErrorException Fuzz.check(IdDict(Fuzz.IdentityKey(1) => 1, Fuzz.IdentityKey(1) => 2), ids)
        # Reject lost aliases, merged aliases and reused originals for big numbers.
        number, equal_number = big(1), big(1)
        @test Fuzz.check((number, number), (big(1), big(1))) !== nothing
        @test Fuzz.check((number, equal_number), let c = big(1); (c, c) end) !== nothing
        @test Fuzz.check((number, equal_number), (equal_number, big(1))) !== nothing
        @test Fuzz.check((number, number), let c = big(1); (c, c) end) === nothing
        # The fuzz oracle's lazy wrapper copy must handle Ref-first, boxed-callable and state-first cycles.
        gain = LawGain(nothing, 3)
        boxed = FunctionWrapper{Int, Tuple{Int}}(BoxedLaw(gain, [1]))
        gain.wrapper = boxed
        aware = FunctionWrapper{Float64, Tuple{Float64}}(SelfAwareLaw(nothing))
        aware.obj[].wrapper = aware
        stateful = FunctionWrapper{Int, Tuple{Int}}(Fuzz.StateLaw(3))
        stateful.obj[].wrapper = stateful
        for value in ((boxed.obj, boxed), (aware.obj, aware), (aware.obj[], aware), (stateful.obj[], stateful), (stateful, stateful.obj[]))
            copied = deepcopy(value)
            @test Fuzz.check(value, copied; bits_fields = false) === nothing
            GC.gc()
            wrapper, copied_wrapper = value[1] isa FunctionWrapper ? (value[1], copied[1]) : (value[2], copied[2])
            @test copied_wrapper(2) == wrapper(2)
        end
        # Base.deepcopy has matching methods for the types with a method of their own, so both copies
        # must pass for every family. Base.deepcopy reads every field as a value.
        copiers = (("specialized_deepcopy", specialized_deepcopy, true), ("Base.deepcopy", deepcopy, false))
        families = (("plain", FUZZ_FAMILY), ("method", FUZZ_METHOD_FAMILY))
        for (family_name, family) in families, seed in 1:200, (name, copier, bits_fields) in copiers
            report = try
                value = Fuzz.generate(Xoshiro(seed), family)
                Fuzz.check(value, copier(value); bits_fields)
            catch e
                "exception: " * sprint(showerror, e)
            end
            @test report === nothing
            report === nothing || println("fuzz $family_name family, seed $seed, $name: $report")
        end
        for (i, value) in enumerate(FUZZ_TRIM_VALUES), (name, copier, bits_fields) in copiers
            report = Fuzz.check(value, copier(value); bits_fields)
            @test report === nothing
            report === nothing || println("fuzz trim value $i, $name: $report")
        end
    end

    # The explicit shapes and extension contracts of test/fuzz_cases.jl, with fixed seeds.
    @testset "fuzz cases" begin
        for seed in 1:3, case in FuzzCases.cases(Xoshiro(seed))
            reports = case.run()
            isempty(reports) || println("fuzz case $(case.name), seed $seed: ", join(reports, "; "))
            @test isempty(reports)
        end
    end

    @testset "stress runner" begin
        @test HammerTool.settings(["5", "2"]) == (5.0, 2)
        for args in (["0"], ["-1"], ["NaN"], ["Inf"], ["5", "0"])
            @test_throws ArgumentError HammerTool.settings(args)
        end
        @test_throws ArgumentError HammerTool.reproduce(1, "unknown")
        @test HammerTool.reproduce_arguments(["7", "42"]) == (7, 42)
        @test HammerTool.reproduce_arguments(["7", "method", "42"]) == (7, "method", 42)
        # A worker that ends early, ends by a signal, or exits with a status that does not match its
        # FAIL lines has crashed. A worker without a FAMILY line did not start.
        family, done = "FAMILY round=1 family=plain layouts=8", "DONE round=1 failures=0"
        fail = "FAIL round=1 family=plain seed=2 rng=0x01 copy: at the root: family=trim seed=9"
        @test HammerTool.completed([done], 0, 0) && HammerTool.completed([fail, done], 1, 0)
        @test !HammerTool.completed([family], 0, 0) && !HammerTool.completed([done], 0, 9)
        @test !HammerTool.completed([done], 1, 0) && !HammerTool.completed([fail, done], 0, 0)
        @test HammerTool.outcome([family, done], 0, 0) == "ok" && HammerTool.outcome([family, fail, done], 1, 0) == "failed"
        @test HammerTool.outcome([family], 0, 9) == "crash" && HammerTool.outcome(String[], 1, 0) == "infrastructure"
        @test HammerTool.outcome([family], 0, 0; timed_out = true) == "timeout"
        @test HammerTool.outcome([family], 0, 0; interrupted = true) == "interrupted"
        # The fields of a FAIL line come from its prefix, not from its message.
        @test HammerTool.fields(fail)["family"] == "plain" && HammerTool.fields(fail)["seed"] == "2"
        # Exact seed counts below the 25 seeds between PROGRESS lines: from COMPLETED lines, and for a
        # family that did not complete, the seeds before the case in the case file.
        lines = [family, "COMPLETED round=1 family=plain seeds=7", "FAMILY round=1 family=method layouts=9"]
        @test HammerTool.completed_seeds(lines, "round=1 family=method seed=4 rng=0x02 stage=copy") == Dict("plain" => 7, "method" => 3)
        # A PROGRESS line confirms the case in the record when the worker ended before the next record.
        progress = [family, "PROGRESS round=1 family=plain seed=25"]
        @test HammerTool.completed_seeds(progress, "round=1 family=plain seed=25 rng=0x02 stage=wrapper calls after GC") == Dict("plain" => 25)
        # A log that shows a failure stays, also when the driver stopped its worker.
        @test HammerTool.keep_log("interrupted", [family, fail]) && HammerTool.keep_log("crash", [family])
        @test !HammerTool.keep_log("interrupted", [family]) && !HammerTool.keep_log("ok", [family, done])
        # A replay sets the compile mode of its run, whatever the mode of the process that starts it.
        @test HammerTool.replay_flag("compiled") == `--compile=yes` && HammerTool.replay_flag("interpreted") == `--compile=min`
        # A run starts only through the script, which copies the source for replays.
        @test_throws ArgumentError HammerTool.hammer(["5", "2"])
        # The explicit cases of a round, one at a time.
        @test HammerTool.reproduce(1, "cases", 2) == 0
        @test_throws ArgumentError HammerTool.reproduce(1, "cases", 999)
        # An exception ends a case with its stage and backtrace; here generation finds no types.
        report = only(HammerTool.exercise(DataType[], true, 1, 1))
        @test startswith(report, "exception in stage generate: ArgumentError") && occursin("Stacktrace", report)
        # Both output streams of a worker reach its log, also after the driver closes its handle.
        log = tempname()
        code = "sleep(0.5); print(stdout, \"out\\n\"); flush(stdout); print(stderr, \"err\\n\")"
        wait(HammerTool.launch(`$(Base.julia_cmd()) --startup-file=no -e $code`, log))
        @test readlines(log) == ["out", "err"]
        # The watchdog stops a worker past its limit, by force if SIGTERM does not end it, and the
        # driver can collect it. The sleeper ignores SIGTERM, except on Windows, where kill ends a
        # process at once. It prints `ready` when it is ready to be stopped.
        sleeper_log = tempname()
        sleep_code = "println(\"ready\"); flush(stdout); sleep(600)"
        sleep_command = Sys.iswindows() ? `$(Base.julia_cmd()) --startup-file=no -e $sleep_code` : `sh -c 'trap "" TERM; echo ready; exec sleep 600'`
        sleeper = HammerTool.launch(sleep_command, sleeper_log)
        @test timedwait(() -> occursin("ready", read(sleeper_log, String)), 60) === :ok
        start = time_ns()
        @test !HammerTool.overdue(start, 60) && HammerTool.overdue(start - UInt64(61e9), 60)
        HammerTool.stop!([sleeper])
        @test process_exited(sleeper) && !success(sleeper)
        @test Sys.iswindows() || sleeper.termsignal == Base.SIGKILL
        # The checks of a worker for one seed of each family, and for one seed of one family only.
        @test HammerTool.reproduce(1, 1) == 0
        @test HammerTool.reproduce(1, "plain", 3) == 0
        # The replay of a run requires the package from the run's source.
        @test HammerTool.hammer(["replay", pkgdir(SpecializedDeepcopy), "7", "1", "plain", "1"]) == 0
        @test redirect_stdout(() -> HammerTool.hammer(["replay", tempdir(), "7", "1", "plain", "1"]), devnull) == 1
        # A worker reports the exact count of each family and of the explicit cases.
        output = tempname()
        status = open(io -> redirect_stdout(() -> HammerTool.worker(1, 0.0, mktempdir(), UInt(7)), io), output, "w")
        lines = readlines(output)
        @test status == 0 && last(lines) == "DONE round=1 failures=0"
        expected = Dict("plain" => 0, "method" => 0, "trim" => 0, "cases" => length(FuzzCases.cases(Xoshiro(1))))
        @test HammerTool.completed_seeds(lines, "") == expected
        # The failed check of an explicit case reaches the log of a worker or a replay before the next
        # case starts, so Ctrl-C in a later case keeps it. The second case reads the log.
        stub = StubbedHammerTool.FuzzCases
        first_failure = r"^FAIL round=1 family=cases seed=1 "m
        stub.LOG[], stub.SEEN[] = tempname(), ""
        @test_throws InterruptException open(io -> redirect_stdout(() -> StubbedHammerTool.worker(1, 0.0, mktempdir(), UInt(7)), io), stub.LOG[], "w")
        @test occursin(first_failure, stub.SEEN[])
        stub.LOG[], stub.SEEN[] = tempname(), ""
        @test open(io -> redirect_stdout(() -> StubbedHammerTool.reproduce(1, "cases"), io), stub.LOG[], "w") == 1
        @test occursin(first_failure, stub.SEEN[])
        # Ctrl-C stops the active worker, which is not a crash, and the driver still writes its summary.
        # Rounds count from 1 in each run, and a run leaves the files of an earlier run alone.
        parent = mktempdir()
        earlier = mkpath(joinpath(parent, "run-earlier"))
        write(joinpath(earlier, "round-7.log"), "FAIL round=7 family=plain seed=3 rng=0x03 copy: at the root\n")
        run = joinpath(parent, "run-later")
        status = redirect_stdout(devnull) do
            task = @async HammerTool.driver(1, 1; run)
            # Once the round log exists, the driver task waits in `sleep`, where Ctrl-C reaches it.
            timedwait(() -> isfile(joinpath(run, "round-1.log")), 60)
            schedule(task, InterruptException(); error = true)
            fetch(task)
        end
        @test status == 1
        @test readdir(run) == ["environment", "hammer.log", "run.toml"]
        @test isfile(joinpath(earlier, "round-7.log"))
        summary = readlines(joinpath(run, "hammer.log"))
        @test any(startswith("hammer: interrupted after 0 min, 1 rounds, 0 cases"), summary)
        # A run that checked no case does not pass.
        @test last(summary) == "hammer: no case completed"
        @test occursin("status = \"interrupted\"", read(joinpath(run, "run.toml"), String))
    end

    # Check ordinary copying for every trim fixture, including expected trim failures.
    @testset "trim test values: $name" for name in filter(n -> occursin(r"^(failing_)?shape_", String(n)), names(@__MODULE__; all = true))
        value = getfield(@__MODULE__, name)()
        @test typeof(specialized_deepcopy(value)) === typeof(value)
    end

    # Run Aqua and JET in their own environment from test/qa/Project.toml.
    # JET does not support Julia prereleases.
    if isempty(VERSION.prerelease)
        @testset "Aqua and JET" begin
            env = mktempdir()
            # A new file, because `cp` keeps the mode of an installed, read-only package file and
            # Pkg must write the project.
            write(joinpath(env, "Project.toml"), read(joinpath(@__DIR__, "qa", "Project.toml")))
            code = """
                import Pkg
                Pkg.develop(path = $(repr(pkgdir(SpecializedDeepcopy))); io = devnull)
                Pkg.instantiate(; io = devnull)
                include($(repr(joinpath(@__DIR__, "qa", "qa.jl"))))
                """
            # Pkg.test limits the load path of this process and checks bounds. The QA environment needs
            # the default load path, and its packages keep their usual compiled caches.
            variables = filter(((name, _),) -> name ∉ ("JULIA_LOAD_PATH", "JULIA_PROJECT"), ENV)
            julia = joinpath(Sys.BINDIR, Base.julia_exename())
            @test success(pipeline(setenv(`$julia --startup-file=no --project=$env -e $code`, variables); stdout, stderr))
        end
    end

    if VERSION >= v"1.12"
        @testset "juliac --trim=safe" begin
            env = mktempdir()
            out = mktempdir()
            package = pkgdir(SpecializedDeepcopy)
            write(joinpath(env, "Project.toml"), """
                [deps]
                BlockArrays = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
                FunctionWrappers = "069b7b12-0de2-55c6-9aab-29f3d0a68a2e"
                JuliaC = "acedd4c2-ced6-4a15-accc-2607eb759ba2"
                PreallocationTools = "d236fae5-4411-538c-8e31-a6e3d9e00b46"
                SpecializedDeepcopy = "$(Base.PkgId(SpecializedDeepcopy).uuid)"

                [compat]
                BlockArrays = "1"
                FunctionWrappers = "1.1.2"
                JuliaC = "0.3"
                PreallocationTools = "1"

                [sources]
                SpecializedDeepcopy = {path = $(repr(package))}
                """)
            log = joinpath(out, "build.log")
            # Pkg.test limits the load path of this process. The build needs the default one.
            variables = filter(((name, _),) -> name ∉ ("JULIA_LOAD_PATH", "JULIA_PROJECT"), ENV)
            # Keep diagnostic matching independent of terminal colors.
            build = setenv(`$(Base.julia_cmd()) --startup-file=no --color=no --project=$env $(joinpath(@__DIR__, "trim", "build.jl")) $out`, variables)
            built = open(io -> success(pipeline(build; stdout = io, stderr = io)), log, "w")
            built || print(read(log, String))
            @test built

            # JuliaC adds .exe to the executable on Windows.
            executable = joinpath(out, "program", "bin", "specialized_deepcopy_trim" * (Sys.iswindows() ? ".exe" : ""))
            @test isfile(executable)
            if isfile(executable)
                lines = readlines(`$executable`)
                foreach(println, filter(line -> !startswith(line, "ok"), lines))
                @test !isempty(lines) && last(lines) == "done"
                @test all(startswith("ok"), lines[1:(end - 1)])
            elseif built
                print(read(joinpath(out, "program", "build.log"), String))
            end

            results = joinpath(out, "shapes", "results.txt")
            outcomes = isfile(results) ? split.(readlines(results)) : Vector{SubString{String}}[]
            @test length(outcomes) == count(n -> occursin(r"^(failing_)?shape_", String(n)), names(@__MODULE__; all = true))
            # Julia 1.14 accepts the depth-limited call; dynamic-dispatch cases still fail.
            compiled_anyway = VERSION >= v"1.14-" ? ("failing_shape_nesting_beyond_depth_limit",) : ()
            @testset "$name" for (name, outcome) in outcomes
                expected = startswith(name, "failing_") && name ∉ compiled_anyway ? "rejected" : "compiled"
                outcome == expected || print(read(joinpath(out, "shapes", name, "build.log"), String))
                @test outcome == expected
            end
        end
    end
end
