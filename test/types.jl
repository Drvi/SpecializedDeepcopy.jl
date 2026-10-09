# Fixtures shared by the ordinary and trimmed tests.
using SpecializedDeepcopy

mutable struct Owner
    id::Float64
end

struct Block
    weights::Matrix{Float64}
end

mutable struct Record{T}
    value::T
end

mutable struct Node
    label::String
    next::Union{Nothing, Node}
end

# A cycle through a const field: a group can contain itself.
mutable struct Group
    const members::Vector{Group}
end

mutable struct Counter
    const name::String
    const owner::Owner
    @atomic count::Int
end

# Inner constructors that leave the last fields undefined.
mutable struct Lazy
    id::Int
    cache::Vector{Float64}
    Lazy(id) = new(id)
end

struct Prefix
    id::Int
    first::Vector{Float64}
    second::Vector{Float64}
    Prefix(id) = new(id)
    Prefix(id, first) = new(id, first)
    Prefix(id, first, second) = new(id, first, second)
end

# An inline tuple can be undefined: `isdefined` checks the reference stored inside it.
mutable struct LateTuple
    parts::Tuple{Int, Vector{Float64}}
    LateTuple() = new()
    LateTuple(parts) = new(parts)
end

# Reject pointers even inside an inline Handle field.
struct Handle
    address::Ptr{Cvoid}
end
struct Device
    handle::Handle
    gain::Float64
end
mutable struct RawBuffer
    data::Ptr{UInt8}
end
struct Name
    text::Cstring
end
# The Nothing alternative copies; the pointer alternative is rejected.
struct MaybePointer
    address::Union{Nothing, Ptr{Cvoid}}
end

# Sharing the handle explicitly allows SharedDevice to copy.
struct SharedHandle
    address::Ptr{Cvoid}
end
SpecializedDeepcopy.specialized_deepcopy(x::SharedHandle, copies) = x
struct SharedDevice
    handle::SharedHandle
    gain::Float64
end

# A bits-type vector over borrowed memory. Copying one is rejected; SharedVector's method shares it.
struct BorrowedVector <: AbstractVector{Float64}
    address::Ptr{Float64}
    n::Int
end
struct SharedVector <: AbstractVector{Float64}
    address::Ptr{Float64}
    n::Int
end
SpecializedDeepcopy.specialized_deepcopy(x::SharedVector, copies) = x
Base.size(x::Union{BorrowedVector, SharedVector}) = (x.n,)
Base.getindex(x::Union{BorrowedVector, SharedVector}, i::Int) = (checkbounds(x, i); unsafe_load(x.address, i))

# A bits-type key that refers to its bytes through a pointer. Its method duplicates the bytes, so the
# copy of a key has another pointer and another hash. KEY_BYTES keeps all the bytes alive.
struct BytesKey
    address::Ptr{UInt8}
end
const KEY_BYTES = Dict{Ptr{UInt8}, Vector{UInt8}}()
BytesKey(bytes::Vector{UInt8}) = (KEY_BYTES[pointer(bytes)] = bytes; BytesKey(pointer(bytes)))
SpecializedDeepcopy.specialized_deepcopy(x::BytesKey, copies) =
    get!(() -> BytesKey(copy(KEY_BYTES[x.address])), copies, x)::BytesKey

# Duplicate C storage and give each owner its own finalizer.
mutable struct OwnedBuffer
    data::Ptr{UInt8}
    length::Int
    function OwnedBuffer(length::Int)
        buffer = new(Ptr{UInt8}(Libc.malloc(length)), length)
        finalizer(b -> Libc.free(b.data), buffer)
        return buffer
    end
end
function SpecializedDeepcopy.specialized_deepcopy(x::OwnedBuffer, copies)
    haskey(copies, x) && return copies[x]::OwnedBuffer
    y = OwnedBuffer(x.length)
    GC.@preserve x y unsafe_copyto!(y.data, x.data, x.length)
    copies[x] = y
    return y
end

# Rebuild keys derived from object identity.
struct OwnerIndex
    owners::Vector{Owner}
    slots::Dict{UInt, Int}
end
OwnerIndex(owners) = OwnerIndex(owners, Dict(objectid(o) => i for (i, o) in pairs(owners)))
SpecializedDeepcopy.specialized_deepcopy(x::OwnerIndex, copies) =
    OwnerIndex(specialized_deepcopy(x.owners, copies))

# A type assertion makes this abstract field usable under trim.
struct Registry
    owner::Any
end
SpecializedDeepcopy.specialized_deepcopy(x::Registry, copies) =
    Registry(specialized_deepcopy(x.owner::Owner, copies))

# Two windows must retain one shared Storage in the copy.
mutable struct Storage
    data::Vector{Float64}
end
struct Window <: AbstractVector{Float64}
    storage::Storage
    range::UnitRange{Int}
end
Base.size(w::Window) = (length(w.range),)
Base.getindex(w::Window, i::Int) = w.storage.data[w.range[i]]
Base.setindex!(w::Window, v, i::Int) = (w.storage.data[w.range[i]] = v)

# Copying stored objectids leaves stale keys. RebuiltIdentityTable reinserts the copied keys.
abstract type AbstractIdentityTable{K, V} <: AbstractDict{K, V} end
mutable struct IdentityTable{K, V} <: AbstractIdentityTable{K, V}
    slots::Dict{UInt, Int}
    keys::Vector{K}
    vals::Vector{V}
end
mutable struct RebuiltIdentityTable{K, V} <: AbstractIdentityTable{K, V}
    slots::Dict{UInt, Int}
    keys::Vector{K}
    vals::Vector{V}
end
(::Type{T})() where {K, V, T <: AbstractIdentityTable{K, V}} = T(Dict{UInt, Int}(), K[], V[])
function Base.setindex!(t::AbstractIdentityTable, v, k)
    i = get(t.slots, objectid(k), 0)
    if i == 0
        push!(t.keys, k)
        push!(t.vals, v)
        t.slots[objectid(k)] = length(t.keys)
    else
        t.vals[i] = v
    end
    return t
end
Base.get(t::AbstractIdentityTable, k, default) = (i = get(t.slots, objectid(k), 0); i == 0 ? default : t.vals[i])
Base.haskey(t::AbstractIdentityTable, k) = haskey(t.slots, objectid(k))
Base.length(t::AbstractIdentityTable) = length(t.keys)
Base.iterate(t::AbstractIdentityTable, i = 1) = i > length(t.keys) ? nothing : (t.keys[i] => t.vals[i], i + 1)
function SpecializedDeepcopy.specialized_deepcopy(x::RebuiltIdentityTable{K, V}, copies) where {K, V}
    haskey(copies, x) && return copies[x]::RebuiltIdentityTable{K, V}
    y = RebuiltIdentityTable{K, V}()
    copies[x] = y
    for (k, v) in x
        y[specialized_deepcopy(k, copies)] = specialized_deepcopy(v, copies)
    end
    return y
end

# Copying a key's own table hashes the unfinished key using the original owner.
# OwnedKey copies the owner first, so insertion uses the final hash.
mutable struct StaleHashKey
    table::Dict{StaleHashKey, Int}
    owner::Vector{Int}
end
Base.hash(k::StaleHashKey, h::UInt) = hash(objectid(k.owner), h)
mutable struct OwnedKey
    table::Dict{OwnedKey, Int}
    owner::Vector{Int}
end
Base.hash(k::OwnedKey, h::UInt) = hash(objectid(k.owner), h)
function SpecializedDeepcopy.specialized_deepcopy(x::OwnedKey, copies)
    haskey(copies, x) && return copies[x]::OwnedKey
    y = OwnedKey(Dict{OwnedKey, Int}(), specialized_deepcopy(x.owner, copies))
    copies[x] = y
    for (k, v) in x.table
        y.table[specialized_deepcopy(k, copies)] = v
    end
    return y
end

# A shallow copy can retain a pointer after the original's finalizer frees its allocation.
mutable struct MallocBuffer
    data::Ptr{UInt8}
    length::Int
    function MallocBuffer(length::Int)
        buffer = new(Ptr{UInt8}(Libc.malloc(length)), length)
        finalizer(b -> Libc.free(b.data), buffer)
        return buffer
    end
end

# Duplicate the C string with strdup; each copy frees its own allocation.
mutable struct CText
    text::Ptr{UInt8}
    function CText(text::Ptr{UInt8})
        owned = new(ccall(:strdup, Ptr{UInt8}, (Ptr{UInt8},), text))
        finalizer(t -> Libc.free(t.text), owned)
        return owned
    end
end
CText(s::String) = GC.@preserve s CText(pointer(s))
SpecializedDeepcopy.specialized_deepcopy(x::CText, copies) =
    haskey(copies, x) ? copies[x]::CText : (copies[x] = CText(x.text))

# ByteView's pointer targets its own array. OwnedByteView redirects it into the copied array.
struct ByteView
    bytes::Vector{UInt8}
    cursor::Ptr{UInt8}
end
struct OwnedByteView
    bytes::Vector{UInt8}
    cursor::Ptr{UInt8}
end
function SpecializedDeepcopy.specialized_deepcopy(x::OwnedByteView, copies)
    bytes = specialized_deepcopy(x.bytes, copies)
    return OwnedByteView(bytes, pointer(bytes) + (x.cursor - pointer(x.bytes)))
end

# Generic copying keeps a ticket's number. NumberedTicket's override runs the numbering constructor.
const TICKET_COUNT = Ref(0)
mutable struct Ticket
    number::Int
    Ticket() = new(TICKET_COUNT[] += 1)
end
mutable struct NumberedTicket
    number::Int
    NumberedTicket() = new(TICKET_COUNT[] += 1)
end
SpecializedDeepcopy.specialized_deepcopy(x::NumberedTicket, copies) =
    haskey(copies, x) ? copies[x]::NumberedTicket : (copies[x] = NumberedTicket())

# Explicitly share the IOStream and its C handle.
struct Logger
    io::IOStream
    level::Int
end
SpecializedDeepcopy.specialized_deepcopy(x::Logger, copies) = Logger(x.io, x.level)

# Discard cached results instead of copying weak references.
struct ResultCache
    results::WeakKeyDict{Owner, Vector{Float64}}
end
SpecializedDeepcopy.specialized_deepcopy(x::ResultCache, copies) = ResultCache(WeakKeyDict{Owner, Vector{Float64}}())

# A mutable type without fields: each copy is a new object.
mutable struct Token end

# Const bits, Union and reference fields require jl_set_nth_field rather than setfield!.
mutable struct Fixed
    const id::Int
    const tag::Union{Nothing, Int}
    const weights::Vector{Float64}
end

# A const field and an atomic field that the constructor leaves undefined.
mutable struct ConstLater
    const weights::Vector{Float64}
    ConstLater() = new()
end
mutable struct AtomicLater
    @atomic weights::Vector{Float64}
    AtomicLater() = new()
end

# An unset inline Union still contains bits, unlike a reference field.
mutable struct UnionTail
    id::Int
    tag::Union{Nothing, Int}
    UnionTail(id) = new(id)
end
# An atomic Union is stored as a reference and can be undefined.
mutable struct AtomicUnionTail
    id::Int
    @atomic tag::Union{Nothing, Int}
    AtomicUnionTail(id) = new(id)
end

# Contrast a concrete Union with one containing an abstract member.
mutable struct ThreeWay
    value::Union{Nothing, Int, Owner}
end
struct OptionalVector
    data::Union{Nothing, AbstractVector{Float64}}
end
# Five alternatives exceed inference's default Union split limit.
mutable struct FiveWay
    value::Union{Nothing, Int, Float64, String, Owner}
end
# 2^63 tuple variants exceed both the split limit and a signed 64-bit count.
struct Wide
    values::Tuple{ntuple(_ -> Union{Missing, Int}, 63)...}
end

# An uninhabited field must stay undefined.
struct Never
    value::Union{}
    Never() = new()
end
mutable struct MutableNever
    value::Union{}
    MutableNever() = new()
end
# Julia 1.13 inference fails when `isdefined` checks the optional reference, then the uninhabited field.
struct RefThenNever
    id::Int
    weights::Vector{Float64}
    value::Union{}
    RefThenNever(id) = new(id)
    RefThenNever(id, weights) = new(id, weights)
end
mutable struct MutableRefThenNever
    id::Int
    weights::Vector{Float64}
    value::Union{}
    MutableRefThenNever(id) = new(id)
    MutableRefThenNever(id, weights) = new(id, weights)
end

# These unset fields can contain invalid isbits Union selectors: in atomic storage, or in an
# array of immutable structs on Julia 1.10.
const UnionPair = @NamedTuple{a::Union{Nothing, Int}, b::Union{Int8, UInt8}}
mutable struct AtomicPair
    id::Int
    @atomic pair::UnionPair
    AtomicPair(id) = new(id)
    AtomicPair(id, pair) = new(id, pair)
end
struct UnsetPair
    id::Int
    pair::UnionPair
    UnsetPair(id) = new(id)
end

# Recursive Union fields may end in `nothing` or a raw pointer; only the latter is rejected.
# The mutable fixture also exercises a const field.
mutable struct Tricky
    ref::Base.RefValue{Int}
    t::Union{Ptr{UInt}, Tricky, Nothing}
    const cref::Base.RefValue{Int}
end
struct ImmutableTricky
    ref::Base.RefValue{Int}
    t::Union{Ptr{UInt}, ImmutableTricky, Nothing}
    cref::Base.RefValue{Int}
end
# `N` levels of `T` around `tail`. The two Refs of a level are one object.
@generated tricky_chain(::Type{T}, tail, ::Val{N}) where {T, N} =
    foldl((ex, i) -> :(let r = Ref($i); T(r, $ex, r) end), 1:N; init = :tail)

# Mutual recursion through Unions with raw pointer members; the type parameter breaks the definition
# cycle. A chain ends in `nothing`, a pointer, a C string, or a link back to its start.
mutable struct Ping{T}
    const id::Int
    next::Union{Nothing, Ptr{Cvoid}, T}
end
mutable struct Pong
    back::Union{Ping{Pong}, Cstring}
end
# `n` pings around `tail`, with a pong between each pair.
ping_chain(n, tail) = foldl((x, i) -> Ping{Pong}(i, Pong(x)), 2:n; init = Ping{Pong}(1, tail))

# The override also runs inside containers; a Tagged field that constructors can leave unset is copied by bits.
struct Tagged
    value::Union{Int, Float64}
end
SpecializedDeepcopy.specialized_deepcopy(x::Tagged, copies) = Tagged(x.value + 1)
struct TaggedHolder
    tag::Tagged
end
struct LateTagged
    id::Int
    tag::Tagged
    LateTagged(id) = new(id)
    LateTagged(id, tag) = new(id, tag)
end

# The override must also apply when the lock belongs to a condition.
mutable struct SharedLock <: Base.AbstractLock end
SpecializedDeepcopy.specialized_deepcopy(x::SharedLock, copies) = x

# Preserve a condition-lock cycle; Base's lock-copy method would instead call BackLock().
mutable struct BackLock <: Base.AbstractLock
    condition::Any
end
function SpecializedDeepcopy.specialized_deepcopy(x::BackLock, copies)
    haskey(copies, x) && return copies[x]::BackLock
    y = BackLock(nothing)
    copies[x] = y
    y.condition = specialized_deepcopy(x.condition, copies)
    return y
end

# Valid mutable objects can have defined fields after undefined ones.
mutable struct Gapped
    first::Vector{Int}
    second::Vector{Int}
    Gapped() = new()
end

# The same layouts as OwnerIndex, Logger and ResultCache, without a method.
struct StaleOwnerIndex
    owners::Vector{Owner}
    slots::Dict{UInt, Int}
end
StaleOwnerIndex(owners) = StaleOwnerIndex(owners, Dict(objectid(o) => i for (i, o) in pairs(owners)))
struct PlainLogger
    io::IOStream
    level::Int
end
struct PlainResultCache
    results::WeakKeyDict{Owner, Vector{Float64}}
end

# Share C objects through an override on their owner, without pirating Vector's copy method.
struct PointerTable
    addresses::Vector{Ptr{Cvoid}}
end
SpecializedDeepcopy.specialized_deepcopy(x::PointerTable, copies) = PointerTable(copy(x.addresses))
