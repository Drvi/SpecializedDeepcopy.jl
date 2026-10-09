# Trimmed runtime checks: print "ok" or "FAIL" for each, then "done".
# Build identity-keyed tables at run time so their hashes match the loaded objects.
include(joinpath(@__DIR__, "..", "types.jl"))

report(name, ok::Bool) = println(Core.stdout, (ok ? "ok    " : "FAIL  ") * name)

function @main(ARGS)
    owner = Owner(0.0)
    buffer = [owner, owner]
    numeric = [1.0, 2.0]
    flat = collect(1.0:4.0)
    owners = [Owner(1.0), Owner(2.0)]
    node = Node("loop", nothing)
    node.next = node
    group = Group(Group[])
    push!(group.members, group)
    values = (
        buffer, buffer, [numeric, numeric], flat, reshape(flat, 2, 2), owners, view(owners, 2:2),
        BitVector([true, false]), [Block([1.0 2.0; 3.0 4.0])], Dict("gain" => 2.0), Dict(owner => 1),
        IdDict(owner => 2), Base.IdSet{Owner}([owner]), Union{Nothing, Owner}[nothing, owner],
        Record(Record(Record(owner))), ((owner = owner,), owner), node, group,
        Counter("hits", owner, 3), Lazy(1), SharedDevice(SharedHandle(Ptr{Cvoid}(UInt(0x1000))), 2.0),
        OwnerIndex([owner]), Registry(owner), big(2)^100, ReentrantLock(),
    )
    (a, b, nested, copied_flat, copied_square, copied_owners, copied_view, bits, blocks, gains, keyed,
        ids, set, optional, deep, tuples, copied_node, copied_group, counter, lazy, device, index,
        registry, large, held) = specialized_deepcopy(values)
    copied_owner = a[1]
    copied_owner.id = 1.0
    report("aliases inside one buffer", a[1] === a[2] && owner.id == 0.0)
    report("repeated buffer", a === b !== buffer)
    report("nested numeric arrays", nested[1] === nested[2] !== numeric && nested[1] == numeric)
    copied_flat[1] = 10.0
    report("reshape shares one Memory", copied_square[1, 1] == 10.0 && flat[1] == 1.0)
    report("view of a copied parent", copied_view[1] === copied_owners[2] !== owners[2])
    bits[1] = false
    report("BitVector", !bits[1] && values[8][1])
    blocks[1].weights[1] = 5.0
    report("matrix in a struct", values[9][1].weights[1] == 1.0)
    gains["gain"] = 3.0
    report("Dict with plain keys", values[10]["gain"] == 2.0)
    report("Dict keyed by a copied owner", keyed[copied_owner] == 1 && !haskey(keyed, owner))
    report("IdDict keyed by a copied owner", ids[copied_owner] == 2 && !haskey(ids, owner))
    report("IdSet of a copied owner", copied_owner in set && !(owner in set))
    report("Union elements", optional[1] === nothing && optional[2] === copied_owner)
    report("three nested records", deep.value.value.value === copied_owner)
    report("Tuple and NamedTuple", tuples[1].owner === tuples[2] === copied_owner)
    report("cycle through a field", copied_node.next === copied_node !== node)
    report("cycle through a const field", only(copied_group.members) === copied_group !== group)
    @atomic counter.count += 1
    report("const and atomic fields", counter.owner === copied_owner && (@atomic counter.count) == 4)
    report("undefined field", lazy.id == 1 && !isdefined(lazy, :cache))
    report("method that shares a handle", device === values[21])
    report("method that rebuilds identity state", index.slots[objectid(copied_owner)] == 1)
    report("method that asserts an abstract field", registry.owner === copied_owner)
    report("BigInt", large == values[24] && large !== values[24])
    report("new lock", held !== values[25])
    report("later copy is independent", specialized_deepcopy(buffer)[1] !== copied_owner)

    # Custom pointer copies and low-level field layouts.
    c_buffer = OwnedBuffer(2)
    unsafe_store!(c_buffer.data, 0x07)
    token = Token()
    later = AtomicLater()
    (copied_buffer, copied_text, copied_tokens, fixed, copied_later, three) = specialized_deepcopy(
        (c_buffer, CText("text"), (token, token), Fixed(1, nothing, [2.0]), later, ThreeWay(owner)))
    report("C memory duplicated by a method", copied_buffer.data != c_buffer.data && unsafe_load(copied_buffer.data) == 0x07)
    report("C string duplicated by a method", unsafe_string(copied_text.text) == "text")
    report("mutable type without fields", copied_tokens[1] === copied_tokens[2] !== token)
    report("const bits, Union and reference fields", fixed.id == 1 && fixed.tag === nothing && fixed.weights == [2.0])
    report("undefined atomic field", !isdefined(copied_later, :weights))
    report("field of a Union of three types", three.value isa Owner && three.value !== owner)
    pointer_error = try
        specialized_deepcopy(Device(Handle(Ptr{Cvoid}(UInt(0x1000))), 1.0))
        false
    catch e
        e isa PointerCopyError && e.type === Handle && e.field === :address
    end
    report("raw pointer field", pointer_error)
    chain_error = try
        specialized_deepcopy(tricky_chain(Tricky, Ptr{UInt}(UInt(0x1000)), Val(3)))
        false
    catch e
        e isa PointerCopyError && e.type === Ptr{UInt} && e.field === nothing
    end
    report("raw pointer at the end of a self-nesting chain", chain_error)
    println(Core.stdout, "done")
    return 0
end
