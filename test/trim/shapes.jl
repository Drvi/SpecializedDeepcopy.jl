# Separate --trim=safe builds prevent shared compiled code from hiding missing specializations.
# Trim builds do not link or run these probes; main.jl checks trimmed execution.
# build.jl discovers definitions named `shape_*` or `failing_shape_*`.
# `failing_shape_*` expects rejection, except for version overrides in runtests.jl.
include(joinpath(@__DIR__, "..", "fuzz.jl"))
using BlockArrays: BlockedArray
using FunctionWrappers: FunctionWrapper
using PreallocationTools: DiffCache
using Random: Xoshiro

struct Triple
    parts::Tuple{Owner, Vector{Float64}, Vector{Float64}}
end

struct Loose
    owner::Any
end

struct Labelled
    label::String
    id::Int
end

mutable struct VectorNode
    next::Union{Nothing, Vector{VectorNode}}
end

mutable struct UnionTree
    children::Vector{Union{Nothing, UnionTree}}
end

mutable struct UnionDictTree
    children::Dict{String, Union{Nothing, UnionDictTree}}
end

mutable struct UnionSetTree
    members::Base.IdSet{Union{Owner, UnionSetTree}}
end

# Model MTKParameters.nonnumeric: one vector per declared parameter type, without narrowing it.
abstract type AbstractTable end

# Interpolation data with a mutable lookup cache and optional derivatives.
struct LinearTable <: AbstractTable
    t::Vector{Float64}
    u::Vector{Float64}
    last_index::Base.RefValue{Int}
    derivative::Union{Nothing, Vector{Float64}}
end

struct ConstantTable <: AbstractTable
    value::Float64
end

# Nested data includes an optional unit and a Tuple with a Union element.
struct Medium
    name::String
    data::@NamedTuple{critical::@NamedTuple{p::Float64, T::Float64}, coefficients::Matrix{Float64}, phases::Vector{Symbol}}
    tables::Dict{Symbol, LinearTable}
    unit::Union{Nothing, Symbol}
    limits::Tuple{Float64, Union{Missing, Float64}}
end

# The parent link can recurse to a controller with the same callable type.
mutable struct Controller{F}
    gain::Float64
    law::F
    medium::Union{Nothing, Medium}
    history::Vector{Union{Missing, Float64}}
    active::BitVector
    parent::Union{Nothing, Controller{F}}
end

# Two buffers share a table; the controllers share a callable, and the child refers to its parent.
function nonnumeric_buffers()
    density = LinearTable([0.0, 1.0], [1.0, 2.0], Ref(1), nothing)
    enthalpy = LinearTable([0.0, 1.0], [3.0, 4.0], Ref(1), [1.0, 1.0])
    data = (critical = (p = 5.78e6, T = 351.3), coefficients = [1.0 2.0; 3.0 4.0], phases = [:liquid, :vapor])
    medium = Medium("R32", data, Dict(:density => density, :enthalpy => enthalpy), :SI, (0.0, missing))
    weights = [1.0, 2.0]
    law = x -> weights[1] * x
    parent = Controller(1.0, law, medium, Union{Missing, Float64}[missing, 1.0], trues(2), nothing)
    child = Controller(2.0, law, nothing, Union{Missing, Float64}[], falses(0), parent)
    return ([medium], Union{Nothing, LinearTable}[density, nothing], [parent, child],
        [Dict("a" => [1.0], "b" => [2.0])], [:mode_a, :mode_b], ["label"])
end

# On Julia 1.12–1.13, inference limits can leave calls without compiled code under --trim.
# These probes cover the recursion and declared-type shapes that trigger those limits.

# Lazy's Vector field is not a type parameter; the hook prevents inference from widening it.
shape_struct_with_vector_field() = (Lazy(1),)

# The field tuple is longer than the outer tuple, but copying it is finite.
shape_tuple_field() = (Triple((Owner(0.0), [1.0], [2.0])),)

# A cycle through a const vector field.
shape_self_containing_group() = (let group = Group(Group[]); push!(group.members, group); group end,)

# Repeated calls for this finite nesting must remain specialized.
shape_same_type_name_nesting() = (Record(Record(Record(Owner(0.0)))),)

# Nested arrays repeat the MemoryRef copy with a different element type.
shape_vector_of_vectors() = ([[1.0], [2.0]],)

# Before Julia 1.13, inference's Union splitting applies a hard recursion limit.
# Splitting these fields in generated code keeps their recursive copies compiled.
shape_union_field() = (let node = Node("loop", nothing); node.next = node; node end,)
shape_union_field_with_parametric_member() = (let node = VectorNode(nothing); node.next = [node]; node end,)
# Recursive unions can declare a pointer alternative without holding a pointer value.
shape_union_field_with_pointer_member() = (tricky_chain(Tricky, nothing, Val(4)), tricky_chain(ImmutableTricky, nothing, Val(4)))

# Five alternatives exceed inference's default split over four types; the copy allows MAX_SPLIT_TYPES.
shape_union_field_with_five_members() = (FiveWay(Owner(0.0)),)

# Recursive Union elements in arrays, dictionaries and identity sets.
shape_union_elements() = (let tree = UnionTree(Union{Nothing, UnionTree}[nothing]); push!(tree.children, UnionTree(Union{Nothing, UnionTree}[])); tree end,)
shape_union_dict_values() = (let tree = UnionDictTree(Dict{String, Union{Nothing, UnionDictTree}}()); tree.children["a"] = nothing; tree.children["b"] = tree; tree end,)
shape_union_set_members() = (let tree = UnionSetTree(Base.IdSet{Union{Owner, UnionSetTree}}()); push!(tree.members, Owner(0.0)); push!(tree.members, tree); tree end,)

# A tuple of 16 nested Records compiles on 1.12–1.13; 17 reaches the limit for repeated calls.
# runtests.jl expects both to compile from 1.14.
@generated nested_records(x, ::Val{N}) where {N} = foldl((ex, _) -> :(Record($ex)), 1:N; init = :x)
shape_nesting_at_depth_limit() = (nested_records(Owner(0.0), Val(SpecializedDeepcopy.MAX_INFERRED_TYPE_DEPTH)),)
failing_shape_nesting_beyond_depth_limit() = (nested_records(Owner(0.0), Val(SpecializedDeepcopy.MAX_INFERRED_TYPE_DEPTH + 1)),)

# A field of abstract type needs runtime dispatch.
failing_shape_abstract_field() = (Loose(Owner(0.0)),)

# Combined nonnumeric buffers with concrete and Union element types.
shape_nonnumeric_buffers() = nonnumeric_buffers()

# Tuple covariance gives (0.0, missing) and (1.0, 2.0) distinct concrete types.
# The copy splits their declared Tuple type in fields and container elements.
shape_tuple_with_union_field() = (Medium("R32", (critical = (p = 1.0, T = 1.0), coefficients = zeros(1, 1), phases = Symbol[]), Dict{Symbol, LinearTable}(), nothing, (0.0, missing)),)
shape_tuple_with_union_elements() = (Tuple{Float64, Union{Missing, Float64}}[(0.0, missing), (1.0, 2.0)],)
shape_tuple_with_union_dict_values() = (let d = Dict{String, Tuple{Float64, Union{Missing, Float64}}}(); d["a"] = (0.0, missing); d["b"] = (1.0, 2.0); d end,)
shape_tuple_with_union_set_members() = (let s = Base.IdSet{Tuple{Float64, Union{Missing, Float64}}}(); push!(s, (0.0, missing)); push!(s, (1.0, 2.0)); s end,)

# Concrete captures, a closure field and Base's callable structs.
shape_closure() = (let weights = [1.0], owner = Owner(1.0); x -> weights[1] * x * owner.id end,)
shape_struct_with_closure_field() = (Record(let weights = [1.0]; x -> weights[1] * x end),)
shape_base_function_types() = (Base.Fix1(*, [2.0]), Returns([1.0]), first ∘ Returns([1.0]))

# The wrapper stores its callable reference as Any, so copying it needs runtime dispatch.
failing_shape_function_wrapper() = (FunctionWrapper{Float64, Tuple{Float64}}(x -> 2.0 * x),)

# Float64 buffers resembling ModelingToolkit's discrete portion and scratch cache.
# Construct them at load time to compile copying alone; get_tmp is not exercised here.
const BLOCKED_BUFFER = BlockedArray([1.0, 2.0, 3.0], [2, 1])
const DIFF_CACHE = DiffCache(zeros(3))
shape_blocked_array() = (BLOCKED_BUFFER,)
shape_diffcache() = (DIFF_CACHE,)

# Reassigning a capture stores it in Core.Box, whose field is Any.
function boxed_counter()
    count = 0
    increment() = (count += 1)
    return increment
end
failing_shape_boxed_closure() = (boxed_counter(),)

# Regression cases for nested calls with Any in Base.deepcopy.
shape_struct_holding_mutable_struct() = (Record(Owner(0.0)),)
shape_tuple_of_mutable_structs() = ((Owner(0.0), Owner(1.0)),)
shape_named_tuple_with_mutable_struct() = ((owner = Owner(0.0),),)
shape_struct_with_matrix_field() = ([Block([1.0 2.0; 3.0 4.0])],)
shape_bitvector() = (BitVector([true, false]),)
shape_methods_without_inline() = (Registry(Owner(0.0)), OwnerIndex([Owner(0.0)]), SharedDevice(SharedHandle(Ptr{Cvoid}(UInt(0x1000))), 1.0))

# Baseline cases that Base.deepcopy also handles under --trim.
shape_struct_with_string_field() = ([Labelled("owner", 1)],)
shape_dict_with_plain_values() = (Dict("gain" => 2.0),)
shape_vector_of_union_elements() = (Union{Nothing, Owner}[nothing, Owner(0.0)],)

# Seeded fuzz graphs, including custom copy methods; `trim` restricts the field choices.
# Stored fixtures use integer, String and Symbol keys; mutable identity keys belong in main.jl.
const FUZZ_TRIM_FAMILY = Fuzz.define_types!(Xoshiro(2), 8; trim = true, methods = true)
const FUZZ_TRIM_VALUES = ntuple(i -> Fuzz.generate(Xoshiro(i), FUZZ_TRIM_FAMILY), 3)
shape_fuzz_1() = FUZZ_TRIM_VALUES[1]
shape_fuzz_2() = FUZZ_TRIM_VALUES[2]
shape_fuzz_3() = FUZZ_TRIM_VALUES[3]
