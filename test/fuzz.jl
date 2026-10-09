# Generate struct layouts and graphs, then check their copies for types, values, undefined fields,
# aliases, cycles and mutable isolation, including shared array storage. Base.deepcopy gets the copy
# methods of the fixtures, so that it is an oracle for every family; the method family exercises the
# overrides. test/fuzz_cases.jl adds explicit shapes, and scripts/fuzz_hammer.jl runs both for longer.
module Fuzz

using FunctionWrappers: FunctionWrapper
using Random: AbstractRNG
using SpecializedDeepcopy: SpecializedDeepcopy, specialized_deepcopy

# Parametric types whose declared field types can be abstract, as in `Holder{Real}(1.0)`.
mutable struct Holder{T}
    value::T
end
struct Couple{A, B}
    first::A
    second::B
end

# Overrides that share an object, rebuild a value, or share a condition's lock.
mutable struct Shared
    id::Int
end
SpecializedDeepcopy.specialized_deepcopy(x::Shared, copies) = x
struct Rebuilt{T}
    inner::T
end
SpecializedDeepcopy.specialized_deepcopy(x::Rebuilt{T}, copies) where {T} = Rebuilt{T}(specialized_deepcopy(x.inner, copies))
mutable struct FuzzLock <: Base.AbstractLock end
SpecializedDeepcopy.specialized_deepcopy(x::FuzzLock, copies) = x
const LOCK = FuzzLock()

# A callable that can refer back to its wrapper, and a plain callable whose override must be skipped.
mutable struct Law
    state::Any
end
(::Law)(x::Int) = x
struct PlainLaw
    gain::Int
end
(law::PlainLaw)(x::Int) = law.gain * x
SpecializedDeepcopy.specialized_deepcopy(::PlainLaw, copies) = error("a method was called for a plain value")

# Custom Ref conversion reads state that can point back to the wrapper.
# Other graph roots can make the copy reach this state first.
struct StateLaw
    gain::Int
end
(law::StateLaw)(x::Int) = law.gain * x
mutable struct LawState
    wrapper::Any
    law::Base.RefValue{StateLaw}
end
Base.cconvert(::Type{Ref{StateLaw}}, law::StateLaw) = Ref(LawState(nothing, Ref(law)))
Base.unsafe_convert(::Type{Ref{StateLaw}}, root::Base.RefValue{LawState}) = Base.unsafe_convert(Ref{StateLaw}, root[].law)

# A value without references whose copy method changes it, so that the check sees where the method ran.
# The copy takes the bits of a field that a constructor can leave unset, so there the value stays.
struct Bumped
    value::Union{Nothing, Int}
end
bumped(x::Bumped) = Bumped(x.value === nothing ? nothing : x.value + 1)
SpecializedDeepcopy.specialized_deepcopy(x::Bumped, copies) = bumped(x)

# Dictionary keys with an ID that the copy keeps, so that the checker can pair the entries of a table
# and its copy without the original keys. The keys of one generated value have distinct IDs.
struct Tag
    id::Int
end
# Tuple keys whose copy method changes the String, so that the copied table must reinsert them.
const TaggedKey = Tuple{Tag, String}
const NamedKey = @NamedTuple{tag::Tag, name::String}
retag(k::TaggedKey) = (k[1], "c" * k[2])
retag(k::NamedKey) = (tag = k.tag, name = "c" * k.name)
SpecializedDeepcopy.specialized_deepcopy(k::Union{TaggedKey, NamedKey}, copies) = retag(k)
# A key that hashes by identity.
mutable struct IdentityKey
    const id::Int
end
# A key that hashes by its contents. A change of its owner after insertion would make the original
# table stale, so the generator keeps owners away from rewiring.
mutable struct ContentKey
    const id::Int
    owner::Vector{Int}
end
Base.hash(k::ContentKey, h::UInt) = hash(k.owner, hash(k.id, h))
Base.isequal(a::ContentKey, b::ContentKey) = a.id == b.id && isequal(a.owner, b.owner)
Base.:(==)(a::ContentKey, b::ContentKey) = isequal(a, b)
# A key whose hash reads the identity of a field, in a table that the key refers to. The copied table
# receives the key before the copy sets that field. Base.deepcopy reads the unset field and crashes,
# so only the explicit cases of test/fuzz_cases.jl use this key, without Base.deepcopy.
mutable struct LoopKey
    const id::Int
    table::Dict{LoopKey, Int}
    owner::Vector{Int}
end
Base.hash(k::LoopKey, h::UInt) = hash(objectid(k.owner), hash(k.id, h))
# A key whose hash and equality read whether two of its fields are one object, in a table that the key
# refers to. While the copy sets those fields, two keys can compare equal for a moment. Like LoopKey,
# only the explicit cases use this key, without Base.deepcopy.
mutable struct AliasKey
    const id::Int
    left::Vector{Int}
    table::Dict{AliasKey, Int}
    right::Vector{Int}
end
aliased(k::AliasKey) = k.left === k.right
Base.hash(k::AliasKey, h::UInt) = hash(aliased(k), h)
Base.isequal(a::AliasKey, b::AliasKey) = aliased(a) == aliased(b)
Base.:(==)(a::AliasKey, b::AliasKey) = isequal(a, b)
const FixtureKey = Union{TaggedKey, NamedKey, IdentityKey, ContentKey, LoopKey, AliasKey}
key_id(k) = k
key_id(k::TaggedKey) = (:tagged, k[1].id)
key_id(k::NamedKey) = (:named, k.tag.id)
key_id(k::Union{IdentityKey, ContentKey, LoopKey, AliasKey}) = (nameof(typeof(k)), k.id)

# Give Base.deepcopy matching fixture overrides. The FunctionWrapper method is test-only piracy;
# it always defers both pointers, using FunctionWrappers' initialization for precompiled wrappers.
Base.deepcopy_internal(x::Shared, ::IdDict) = x
Base.deepcopy_internal(x::Rebuilt{T}, d::IdDict) where {T} = Rebuilt{T}(Base.deepcopy_internal(x.inner, d))
Base.deepcopy_internal(x::FuzzLock, ::IdDict) = x
Base.deepcopy_internal(x::Bumped, ::IdDict) = bumped(x)
Base.deepcopy_internal(k::Union{TaggedKey, NamedKey}, ::IdDict) = retag(k)
function Base.deepcopy_internal(x::FunctionWrapper{R, A}, d::IdDict) where {R, A}
    haskey(d, x) && return d[x]::FunctionWrapper{R, A}
    # The pointer fields of the new struct are null.
    y = ccall(:jl_new_struct_uninit, Any, (Any,), FunctionWrapper{R, A})::FunctionWrapper{R, A}
    d[x] = y
    setfield!(y, :obj, Base.deepcopy_internal(getfield(x, :obj), d))
    setfield!(y, :objT, getfield(x, :objT))
    return y
end

const SCALARS = Any[Int, Int8, UInt64, Float64, Float32, Bool, UInt8, Char, Symbol, String, Nothing, Missing]
const REGEXES = [r"a+", r"b*c", r"\d"]
const KEYS = Any[Int, String, Symbol]
# Trim families keep the scalar keys: a stored table with identity keys would need hashes from after
# the build, and their fixed seeds guard known compiler limits.
const ID_KEYS = Any[KEYS; IdentityKey; ContentKey]
const METHOD_KEYS = Any[ID_KEYS; TaggedKey; NamedKey; Union{Nothing, TaggedKey}]
key_types(trim, methods) = trim ? KEYS : methods ? METHOD_KEYS : ID_KEYS
const NAMES = [:red, :green, :blue]
const DEFINED = Ref(0)

wide_tuple(n) = Tuple{ntuple(_ -> Union{Missing, Int}, n)...}

# Self references use an optional field or a container so generation can terminate.
# `trim` restricts dynamic types; `methods` enables fixtures with custom copy behavior.
function random_type(rng, family, self, depth; trim, methods)
    inner() = random_type(rng, family, self, depth - 1; trim, methods)
    leaf() = rand(rng, vcat(SCALARS, family))
    # Wrapping the 16 variants of wide_tuple(4) in another Union would exceed the trim split limit.
    optional(T) = (t = inner(); trim && t === wide_tuple(4) ? t : :(Union{$T, $t}))
    depth <= 0 && return leaf()
    choice = rand(rng, 1:19)
    # A Union with a raw pointer member, which no value holds, must copy like one without it.
    choice == 1 && return rand(rng, Bool) ? :(Union{Nothing, $self}) : :(Union{Nothing, Ptr{UInt}, $self})
    choice == 2 && return :(Vector{$self})
    choice == 3 && return :(Dict{$(rand(rng, key_types(trim, methods))), $self})
    choice == 4 && return optional(Nothing)
    choice == 5 && return optional(Missing)
    choice == 6 && return :(Union{$(leaf()), $(leaf())})
    choice == 7 && return :(Vector{$(inner())})
    choice == 8 && return :(Matrix{$(inner())})
    choice == 9 && return :(Dict{$(rand(rng, key_types(trim, methods))), $(inner())})
    choice == 10 && return :(IdDict{$(rand(rng, key_types(trim, methods))), $(inner())})
    # Concrete tuple elements keep trim cases within the union-expansion limit.
    choice == 11 && return trim ? :(Tuple{$(leaf()), $(leaf())}) : :(Tuple{$(inner()), $(inner())})
    choice == 12 && return :(NamedTuple{(:a, :b), Tuple{$(inner()), $(inner())}})
    choice == 13 && return :(Base.RefValue{$(inner())})
    choice == 14 && return rand(rng, Any[BitVector, Set{Int}, Base.IdSet{Int},
        Tuple{Float64, Union{Missing, Float64}}, :(Couple{$(inner()), $(inner())})])
    if choice == 15
        # Exercise the 16-variant limit, its boundary, and integer overflow in variant counting.
        trim && return wide_tuple(4)
        return rand(rng, Any[wide_tuple(4), wide_tuple(5), wide_tuple(63), Core.SimpleVector])
    end
    if choice == 16
        trim && return :(Holder{$(inner())})
        return rand(rng, Any[Any, Real, AbstractVector{Float64}, Function, Tuple{Vararg{Int}}, Holder{Any}, Holder{Real}])
    end
    if choice == 17 && methods
        # Containers without references must still call Bumped's override.
        bumped_types = Any[Bumped, Tuple{Bumped, Int}, Some{Bumped}]
        trim && return rand(rng, Any[Shared, :(Rebuilt{$(inner())}), bumped_types...])
        return rand(rng, Any[Shared, :(Rebuilt{$(inner())}), bumped_types..., FunctionWrapper{Int, Tuple{Int}}, Base.GenericCondition{FuzzLock}])
    end
    if choice == 18
        choices = Any[Threads.Atomic{Int}, :(Some{$(inner())}), :(Pair{$(inner()), $(inner())}),
            :(NTuple{$(rand(rng, 1:6)), $(trim ? leaf() : inner())}), :(Base.ImmutableDict{Int, $(inner())})]
        # Omit C-backed values and the dynamically typed Box from stored trim fixtures.
        trim || append!(choices, Any[BigInt, Regex, Core.Box])
        return rand(rng, choices)
    end
    # Sample wrapper fields directly too.
    choice == 19 && methods && !trim && return FunctionWrapper{Int, Tuple{Int}}
    return leaf()
end

"""
    define_types!(rng, count; trim = false, methods = false) -> Vector{DataType}

Define `count` mutable or immutable types with one to four fields. Layouts can include const,
atomic, undefined and bottom-type fields, plus references to earlier types or themselves.
`trim` restricts field choices for trim fixtures. `methods` enables custom-copy fixtures,
including FunctionWrappers when `trim` is false; Base.deepcopy gets the same copy methods.
"""
function define_types!(rng, count; trim = false, methods = false)
    family = DataType[]
    for _ in 1:count
        name = Symbol(:Fuzz, DEFINED[] += 1)
        mutable = rand(rng, Bool)
        n = rand(rng, 1:4)
        types = Any[random_type(rng, family, name, 2; trim, methods) for _ in 1:n]
        # Put the uninhabited field last so it cannot leave later inline fields unset.
        rand(rng) < 0.1 && (types[n] = Union{})
        fields = map(1:n) do j
            field = :($(Symbol(:f, j))::$(types[j]))
            kind = mutable ? rand(rng, [:plain, :plain, :const, :atomic]) : :plain
            kind === :const ? Expr(:const, field) : kind === :atomic ? :(@atomic $field) : field
        end
        # Positional constructors cannot initialize past an uninhabited field.
        bottom = findfirst(t -> t === Union{}, types)
        settable = bottom === nothing ? n : bottom - 1
        args = [Symbol(:a, j) for j in 1:settable]
        least = rand(rng, 0:settable)
        constructors = [:($name($(args[1:k]...)) = new($(args[1:k]...))) for k in least:settable]
        Core.eval(@__MODULE__, Expr(:struct, mutable, name, Expr(:block, fields..., constructors...)))
        push!(family, Core.eval(@__MODULE__, name))
    end
    return family
end

mutable struct Env
    rng::AbstractRNG
    family::Vector{DataType}
    pool::Vector{Any}  # candidates for aliases and cycles
    width::Int         # sampled container lengths, column counts and entry counts
    ids::Int           # the last ID given to a fixture key
end

"""
    generate(rng, family; depth = 3, width = 3) -> Tuple

Generate one to three primary roots, then rewire references to introduce aliases and cycles.
May prepend pooled objects, including wrapper state, to vary traversal order.
`depth` steers termination and `width` controls sampled container sizes; fixed layouts and
rewiring mean neither is a strict bound on the final graph.
"""
function generate(rng, family; depth = 3, width = 3)
    env = Env(rng, family, Any[], width, 0)
    roots = Any[value(env, rand(rng, family), depth) for _ in 1:rand(rng, 1:3)]
    rewire!(env, rand(rng, 0:(width + 1)))
    first_roots = Any[]
    states = filter(o -> o isa LawState, env.pool)
    !isempty(states) && rand(rng, Bool) && push!(first_roots, rand(rng, states))
    !isempty(env.pool) && rand(rng) < 0.3 && push!(first_roots, rand(rng, env.pool))
    return Tuple(vcat(first_roots, roots))
end

# Near the depth limit, prefer terminal Union members, empty arrays/dicts and shorter constructors.
function value(env, @nospecialize(F), depth)
    rng = env.rng
    if F isa Union
        # No value holds a raw pointer, whose copy raises PointerCopyError.
        members = filter(m -> !(m <: Ptr), Base.uniontypes(F))
        if depth <= 0
            terminal = filter(m -> isbitstype(m) || m === String || m <: FixtureKey, members)
            isempty(terminal) || (members = terminal)
        end
        return value(env, rand(rng, members), depth)
    elseif F === Any
        choices = depth <= 0 ? SCALARS : vcat(SCALARS, env.family, Any[Vector{Int}, Vector{Any}, Dict{String, Int}])
        return value(env, rand(rng, choices), depth)
    elseif F === Real
        return value(env, rand(rng, Any[Int, Float64]), depth)
    elseif F === AbstractVector{Float64}
        parent = rand(rng, rand(rng, 1:4))
        push!(env.pool, parent)
        return rand(rng, Bool) ? parent : view(parent, 1:(length(parent) - 1))
    elseif F === Function
        rand(rng, Bool) && return sin
        captured = value(env, Vector{Float64}, 1)
        return () -> captured
    elseif F === Tuple{Vararg{Int}}
        return ntuple(_ -> rand(rng, Int), rand(rng, 0:3))
    elseif F === Int || F === Int8 || F === UInt64 || F === UInt8
        return rand(rng, F)
    elseif F === Float64 || F === Float32
        return rand(rng, F)
    elseif F === Bool
        return rand(rng, Bool)
    elseif F === BigInt
        # A BigInt is a mutable object, which rewiring can share.
        x = big(rand(rng, Int))
        push!(env.pool, x)
        return x
    elseif F === Bumped
        return Bumped(rand(rng, Bool) ? nothing : rand(rng, Int))
    elseif F === IdentityKey
        # Rewiring can make other objects refer to the key; its only field is const.
        x = IdentityKey(env.ids += 1)
        push!(env.pool, x)
        return x
    elseif F === ContentKey
        # Neither the key nor its owner joins the pool, so rewiring cannot change the key's hash.
        return ContentKey(env.ids += 1, rand(rng, Int, rand(rng, 0:2)))
    elseif F === TaggedKey
        return (Tag(env.ids += 1), String(rand(rng, 'a':'z', rand(rng, 0:3))))
    elseif F === NamedKey
        return (tag = Tag(env.ids += 1), name = String(rand(rng, 'a':'z', rand(rng, 0:3))))
    elseif F === Regex
        return rand(rng, REGEXES)
    elseif F === Core.Box
        x = Core.Box(value(env, Any, depth - 1))
        push!(env.pool, x)
        return x
    elseif F === Threads.Atomic{Int}
        return Threads.Atomic{Int}(rand(rng, Int))
    elseif F <: Some
        return F(value(env, F.parameters[1], depth - 1))
    elseif F <: Pair
        A, B = F.parameters
        return F(value(env, A, depth - 1), value(env, B, depth - 1))
    elseif F <: Base.ImmutableDict
        K, V = F.parameters
        x = F()
        for _ in 1:(depth <= 0 ? 0 : rand(rng, 0:env.width))
            x = Base.ImmutableDict(x, value(env, K, 0) => value(env, V, depth - 1))
        end
        return x
    elseif F === Char
        return rand(rng, 'a':'z')
    elseif F === Symbol
        return rand(rng, NAMES)
    elseif F === String
        return String(rand(rng, 'a':'z', rand(rng, 0:5)))
    elseif F === Nothing
        return nothing
    elseif F === Missing
        return missing
    elseif F === Core.SimpleVector
        return Core.svec(Any[value(env, Any, depth - 1) for _ in 1:rand(rng, 0:env.width)]...)
    elseif F <: Tuple
        return Tuple(Any[value(env, P, depth) for P in F.parameters])
    elseif F <: NamedTuple
        return F(Tuple(Any[value(env, P, depth) for P in F.parameters[2].parameters]))
    elseif F <: Array
        T = eltype(F)
        n = depth <= 0 ? 0 : rand(rng, 0:env.width)
        x = ndims(F) == 1 ? Vector{T}(undef, n) : Matrix{T}(undef, 2, n)
        for i in eachindex(x)
            # Leave holes in reference-containing arrays.
            (inline_type(T) || rand(rng) < 0.8) && (x[i] = value(env, T, depth - 1))
        end
        # Make shared storage available to later rewiring.
        if ndims(F) == 2 && rand(rng, Bool)
            flat = vec(x)
            push!(env.pool, flat)
            x = reshape(flat, size(x))
        end
        # From Julia 1.11, a vector can share the storage of another one from a later offset.
        if isdefined(Base, :Memory) && ndims(F) == 1 && length(x) >= 2 && rand(rng) < 0.25
            push!(env.pool, x)
            x = shifted(x)
        end
        push!(env.pool, x)
        return x
    elseif F <: AbstractDict
        x = F()
        for _ in 1:(depth <= 0 ? 0 : rand(rng, 0:env.width))
            x[value(env, keytype(F), 0)] = value(env, valtype(F), depth - 1)
        end
        push!(env.pool, x)
        return x
    elseif F <: AbstractSet
        x = F()
        for _ in 1:rand(rng, 0:env.width)
            push!(x, rand(rng, Int))
        end
        return x
    elseif F <: Base.RefValue
        T = eltype(F)
        x = !inline_type(T) && rand(rng) < 0.2 ? F() : F(value(env, T, depth - 1))
        push!(env.pool, x)
        return x
    elseif F === BitVector
        return BitVector(rand(rng, Bool, rand(rng, 0:5)))
    elseif F <: Holder
        x = F(value(env, F.parameters[1], depth - 1))
        push!(env.pool, x)
        return x
    elseif F <: Couple
        A, B = F.parameters
        return F(value(env, A, depth - 1), value(env, B, depth - 1))
    elseif F === Shared
        return Shared(rand(rng, Int))
    elseif F <: Rebuilt
        return F(value(env, F.parameters[1], depth - 1))
    elseif F <: Base.GenericCondition
        return Base.GenericCondition(LOCK)
    elseif F <: FunctionWrapper
        choice = rand(rng, 1:4)
        callable = if choice == 1
            law = Law(value(env, Any, depth - 1))
            push!(env.pool, law)
            law
        elseif choice == 2
            PlainLaw(rand(rng, Int))
        elseif choice == 3
            captured = value(env, Vector{Int}, 1)
            x -> x + length(captured)
        else
            StateLaw(rand(rng, 1:5))
        end
        x = F(callable)
        if callable isa StateLaw
            state = getfield(x, :obj)[]
            state.wrapper = x
            push!(env.pool, state)
        end
        push!(env.pool, x)
        return x
    elseif F in env.family
        return struct_value(env, F, depth)
    end
    error("no generator for $F")
end

function struct_value(env, T, depth)
    rng = env.rng
    if ismutabletype(T) && rand(rng) < 0.25
        existing = filter(o -> o isa T, env.pool)
        isempty(existing) || return rand(rng, existing)
    end
    n = fieldcount(T)
    bottom = findfirst(j -> fieldtype(T, j) === Union{}, 1:n)
    settable = bottom === nothing ? n : bottom - 1
    least = n - T.name.n_uninitialized
    # Initialize inline fields so comparisons never depend on unspecified bits.
    counts = [k for k in least:settable if all(j -> !inline(T, j), (k + 1):n)]
    k = depth <= 0 ? first(counts) : rand(rng, counts)
    x = T(Any[value(env, fieldtype(T, j), depth - 1) for j in 1:k]...)
    ismutabletype(T) && push!(env.pool, x)
    return x
end

# Attempt `count` rewires between compatible objects; leave const fields and wrappers intact.
function rewire!(env, count)
    rng = env.rng
    isempty(env.pool) && return
    for _ in 1:count
        a, b = rand(rng, env.pool), rand(rng, env.pool)
        if a isa Array
            a isa Vector && b isa eltype(a) && push!(a, b)
        elseif a isa AbstractDict
            b isa valtype(a) && (a[value(env, keytype(a), 0)] = b)
        elseif a isa Base.RefValue
            b isa eltype(a) && (a[] = b)
        elseif a isa FunctionWrapper
            continue
        else
            T = typeof(a)
            j = rand(rng, 1:fieldcount(T))
            isconst(T, j) && continue
            b isa fieldtype(T, j) || continue
            setfield!(a, j, b, order(T, j)...)
        end
    end
end

order(T, j) = Base.isfieldatomic(T, j) ? (:sequentially_consistent,) : ()

# Inline storage without references has no undefined marker and may contain invalid Union selectors.
# Always initialize these fields, Ref contents and array elements before comparing copies.
inline_type(F) = isbitstype(F) || Base.isbitsunion(F) ||
    (isconcretetype(F) && !ismutabletype(F) && Base.allocatedinline(F) && Base.datatype_pointerfree(F))
# The same for field `j` of `T`. An atomic Union field is stored as a reference.
inline(T, j) = (F = fieldtype(T, j); inline_type(F) && !(Base.isbitsunion(F) && Base.isfieldatomic(T, j)))

struct Mismatch <: Exception
    message::String
end

where_(path) = isempty(path) ? "at the root" : "at " * join(path, ".")
fail(path, what) = throw(Mismatch(string(where_(path), ": ", what)))

# Types whose values are shared between a value and its copy.
shared_by_design(x) = x isa Symbol || x isa Type || x isa Module || x isa Regex || x isa Shared || x isa FuzzLock

const MemoryTypes = isdefined(Base, :GenericMemory) ? Base.GenericMemory : Union{}

# Track array storage and offsets from Julia 1.11, excluding shared empty Memory singletons.
@static if isdefined(Base, :Memory)
    storage(x::Array) = (mem = getfield(getfield(x, :ref), :mem); isempty(mem) ? nothing : mem)
    offset(x::Array) = Base.memoryrefoffset(getfield(x, :ref))
    # A vector that shares the storage of `x` from its second element on.
    shifted(x::Vector) = Base.wrap(Array, memoryref(getfield(getfield(x, :ref), :mem), 2), (length(x) - 1,))
else
    storage(::Array) = nothing
    offset(::Array) = 1
end

# Optional fields with concrete, inline, immutable types without references are copied by bits.
# The package also excludes raw pointers, absent from fuzz values.
bits_field(T, j) = j > fieldcount(T) - T.name.n_uninitialized &&
    (F = fieldtype(T, j); isconcretetype(F) && !ismutabletype(F) && Base.allocatedinline(F) && Base.datatype_pointerfree(F))

# Set and pre-1.11 IdSet store members in a backing dictionary.
table(x::AbstractSet) = hasfield(typeof(x), :dict) ? getfield(x, :dict) : nothing

"""
    check(x, y; bits_fields = true) -> Union{Nothing, String}

Check generated graphs for type, value, alias and ownership mismatches. Return the first mismatch
or `nothing`. Array Memory and offsets are tracked from Julia 1.11; wrapper pointers are checked
against their conversion roots. Strings and big numbers are compared by value, and big numbers are
tracked by identity; `shared_by_design` values may remain shared. A `Bumped` value must show that its
copy method ran, except in a field that `y` copies by its bits (see `bits_field`): set `bits_fields` to
false for a copy from Base.deepcopy, which has no such fields. Uninitialized inline bits can cause
mismatches, so `generate` initializes them.

A table or set with fixture keys (see `key_id`) is compared by pairing its entries by key ID, so its
keys are compared like its values, also when a copy method changes them. Each key of a copied `Dict`
or `IdDict`, and each member of a copied `IdSet`, must find its own entry. Two keys with one ID in an
original table raise an error, because the checker cannot pair them.
"""
function check(x, y; bits_fields = true)
    forward, backward = IdDict{Any, Any}(), IdDict{Any, Any}()
    try
        compare(x, y, forward, backward, Any[], bits_fields)
    catch e
        e isa Mismatch || rethrow()
        return e.message
    end
    original = mutables(x)
    shared = Any[o for o in mutables(y) if o in original]
    isempty(shared) || return "the copy shares $(length(shared)) mutable objects with the original, such as $(summary(first(shared)))"
    return nothing
end

# Pair the entries of a table, or the members of a set, by key ID. Then compare both keys and values
# through the identity maps, which find split and merged aliases. `entries` gives (key, value) pairs.
function compare_paired(@nospecialize(x), @nospecialize(y), entries, forward, backward, path, bits_fields)
    ids = Set{Any}()
    for (k, _) in entries(x)
        id = key_id(k)
        id in ids && error(where_(path), ": two keys of the original have the ID ", repr(id))
        push!(ids, id)
    end
    copied = Dict{Any, Any}()
    for (k, v) in entries(y)
        id = key_id(k)
        haskey(copied, id) && fail(path, "two keys of the copy have the ID $(repr(id))")
        copied[id] = (k, v)
    end
    for (k, v) in entries(x)
        id = key_id(k)
        entry = get(copied, id, nothing)
        entry === nothing && fail(path, "the key with ID $(repr(id)) is missing")
        compare(k, entry[1], forward, backward, [path; :key; id], bits_fields)
        compare(v, entry[2], forward, backward, [path; id], bits_fields)
    end
end

# Each key of a copied hash table finds its own entry; a key in a stale slot does not.
function check_lookups(@nospecialize(y), path)
    if y isa Union{Dict, IdDict}
        for (k, v) in y
            haskey(y, k) && y[k] === v || fail(path, "the copied key $(summary(k)) with ID $(repr(key_id(k))) does not find its entry")
        end
    elseif y isa Base.IdSet
        for member in y
            member in y || fail(path, "the copied member $(summary(member)) is not found")
        end
    end
end

# The functions of the checker do not specialize on the values. The generated types are new in each
# round, and compiling specializations for them took more than half of the time of a check (300 seeds
# on Julia 1.12).
function compare(@nospecialize(x), @nospecialize(y), forward, backward, path, bits_fields)
    typeof(x) === typeof(y) || fail(path, "type $(typeof(x)) became $(typeof(y))")
    if isbits(x) || shared_by_design(x)
        x === y || fail(path, "$(repr(x)) became $(repr(y))")
        return
    end
    if x isa String
        x == y || fail(path, "$(repr(x)) became $(repr(y))")
        return
    end
    if x isa Bumped
        y === bumped(x) || fail(path, "the copy method of Bumped did not run: $(repr(x)) became $(repr(y))")
        return
    end
    if x isa Union{TaggedKey, NamedKey}
        isequal(y, retag(x)) || fail(path, "the copy method of the tuple key did not run: $(repr(x)) became $(repr(y))")
        return
    end
    if x isa Union{BigInt, BigFloat}
        # Compare the numbers and track their identity, without reading their C representation.
        isequal(x, y) || fail(path, "$x became $y")
        seen!(x, y, forward, backward, path)
        return
    end
    if x isa Core.SimpleVector
        # SimpleVector's `===` compares elements rather than storage identity.
        length(x) == length(y) || fail(path, "length $(length(x)) became $(length(y))")
        for i in 1:length(x)
            compare(x[i], y[i], forward, backward, [path; i], bits_fields)
        end
        return
    end
    ismutable(x) && seen!(x, y, forward, backward, path) && return
    if x isa Array || x isa MemoryTypes
        size(x) == size(y) || fail(path, "size $(size(x)) became $(size(y))")
        if x isa Array && storage(x) !== nothing
            offset(x) == offset(y) || fail([path; :mem], "offset $(offset(x)) became $(offset(y))")
            # Compare the whole backing Memory once, including elements outside this array.
            compare(storage(x), storage(y), forward, backward, [path; :mem], bits_fields)
            return
        end
        for i in eachindex(x)
            assigned = isassigned(x, i)
            assigned == isassigned(y, i) || fail([path; i], "assigned $assigned became $(!assigned)")
            assigned && compare(x[i], y[i], forward, backward, [path; i], bits_fields)
        end
    elseif x isa AbstractDict
        length(x) == length(y) || fail(path, "length $(length(x)) became $(length(y))")
        if any(k -> k isa FixtureKey, keys(x))
            compare_paired(x, y, pairs, forward, backward, path, bits_fields)
        else
            # A scalar key is its own copy.
            for (k, v) in x
                haskey(y, k) || fail(path, "the key $(repr(k)) is missing")
                compare(v, y[k], forward, backward, [path; k], bits_fields)
            end
        end
        check_lookups(y, path)
    elseif x isa AbstractSet && table(x) !== nothing
        compare(table(x), table(y), forward, backward, [path; :dict], bits_fields)
    elseif x isa AbstractSet
        length(x) == length(y) || fail(path, "length $(length(x)) became $(length(y))")
        if any(m -> m isa FixtureKey, x)
            compare_paired(x, y, set -> ((m, nothing) for m in set), forward, backward, path, bits_fields)
        else
            for member in x
                member in y || fail(path, "the member $(repr(member)) is missing")
            end
        end
        check_lookups(y, path)
    elseif x isa FunctionWrapper
        getfield(x, :objT) === getfield(y, :objT) || fail(path, "the wrapped type changed")
        compare(getfield(x, :obj), getfield(y, :obj), forward, backward, [path; :obj], bits_fields)
        own = Base.unsafe_convert(Ref{getfield(y, :objT)}, getfield(y, :obj))
        if getfield(y, :ptr) == C_NULL
            # A cold wrapper may already have its object pointer; its first call sets both.
            getfield(y, :objptr) in (C_NULL, own) || fail(path, "the copied wrapper points elsewhere")
        else
            getfield(x, :ptr) == getfield(y, :ptr) || fail(path, "the code pointer changed")
            getfield(y, :objptr) == own || fail(path, "the copied wrapper does not point to its own root")
        end
    else
        T = typeof(x)
        for j in 1:fieldcount(T)
            defined = isdefined(x, j, order(T, j)...)
            defined == isdefined(y, j, order(T, j)...) || fail([path; fieldname(T, j)], "defined $defined became $(!defined)")
            defined || continue
            xj, yj = getfield(x, j, order(T, j)...), getfield(y, j, order(T, j)...)
            if bits_fields && bits_field(T, j)
                xj === yj || fail([path; fieldname(T, j)], "the bits of $(repr(xj)) became $(repr(yj))")
            else
                compare(xj, yj, forward, backward, [path; fieldname(T, j)], bits_fields)
            end
        end
    end
end

# Enforce a one-to-one mapping of mutable objects; return true for a consistent repeat visit.
function seen!(@nospecialize(x), @nospecialize(y), forward, backward, path)
    if haskey(forward, x)
        forward[x] === y || fail(path, "an object that the original shares is not shared by the copy")
        return true
    end
    x === y && fail(path, "the copy is the original object, $(summary(x))")
    haskey(backward, y) && fail(path, "two objects of the original became one object of the copy")
    forward[x] = y
    backward[y] = x
    return false
end

# Collect tracked mutable objects by identity, excluding deliberate sharing and value-only cases.
function mutables(x)
    set = Base.IdSet{Any}()
    collect_mutables!(set, x)
    return set
end

function collect_mutables!(set, @nospecialize(x))
    (isbits(x) || shared_by_design(x) || x isa String) && return
    if x isa Union{BigInt, BigFloat}
        # Track the number as an object without reading its C representation.
        push!(set, x)
        return
    end
    if x isa Core.SimpleVector
        for element in x
            collect_mutables!(set, element)
        end
        return
    end
    if ismutable(x)
        x in set && return
        push!(set, x)
    end
    if x isa Array && storage(x) !== nothing
        # Include elements outside this array's slice of the backing Memory.
        collect_mutables!(set, storage(x))
    elseif x isa Array || x isa MemoryTypes
        for i in eachindex(x)
            isassigned(x, i) && collect_mutables!(set, x[i])
        end
    elseif x isa AbstractDict
        for (k, v) in x
            collect_mutables!(set, k)
            collect_mutables!(set, v)
        end
    elseif x isa AbstractSet && table(x) !== nothing
        collect_mutables!(set, table(x))
    elseif x isa AbstractSet
        for member in x
            collect_mutables!(set, member)
        end
    elseif x isa FunctionWrapper
        collect_mutables!(set, getfield(x, :obj))
    else
        T = typeof(x)
        for j in 1:fieldcount(T)
            isdefined(x, j, order(T, j)...) && collect_mutables!(set, getfield(x, j, order(T, j)...))
        end
    end
end

end # module
