# A type that owns C memory through a raw pointer. Without a method, specialized_deepcopy rejects the
# pointer, and Base.deepcopy copies the address, so the copy uses memory that the original's
# finalizer frees. A method that duplicates the memory gives each copy its own allocation.
#
# Run this file alone from the package directory: julia --project test/examples/externally_allocated_memory.jl
# The test suite includes this file and calls test_externally_allocated_memory(). The trimmed test program
# includes it for its definitions only.
using SpecializedDeepcopy
using Test

# Owns `length` bytes of C memory and frees them in its finalizer.
mutable struct MallocBuffer
    data::Ptr{UInt8}
    length::Int
    function MallocBuffer(length::Int)
        buffer = new(Ptr{UInt8}(Libc.malloc(length)), length)
        finalizer(b -> Libc.free(b.data), buffer)
        return buffer
    end
end

# The same layout, with a method that allocates new memory, copies the bytes while both owners stay
# alive, and gives the copy its own finalizer through the constructor.
mutable struct ExternBuffer
    data::Ptr{UInt8}
    length::Int
    function ExternBuffer(length::Int)
        buffer = new(Ptr{UInt8}(Libc.malloc(length)), length)
        finalizer(b -> Libc.free(b.data), buffer)
        return buffer
    end
end
function SpecializedDeepcopy.specialized_deepcopy_internal(x::ExternBuffer, copies)
    haskey(copies, x) && return copies[x]::ExternBuffer
    y = ExternBuffer(x.length)
    GC.@preserve x y unsafe_copyto!(y.data, x.data, x.length)
    copies[x] = y
    return y
end

function test_externally_allocated_memory()
    @testset "externally allocated memory" begin
        # Without a method: Base.deepcopy keeps the address, and specialized_deepcopy rejects it.
        buffer = MallocBuffer(4)
        unsafe_store!(buffer.data, 0x07)
        @test deepcopy(buffer).data == buffer.data
        @test_throws PointerCopyError(MallocBuffer, :data) specialized_deepcopy(buffer)
        @test unsafe_load(buffer.data) == 0x07

        # With the method: the copy has its own memory, which outlives the original's finalizer.
        owned = ExternBuffer(4)
        unsafe_store!(owned.data, 0x07)
        copied = specialized_deepcopy(owned)
        GC.gc()
        unsafe_store!(copied.data, 0x09)
        @test copied.data != owned.data && unsafe_load(owned.data) == 0x07
        finalize(owned)
        @test unsafe_load(copied.data) == 0x09
        finalize(copied)
    end
end

abspath(PROGRAM_FILE) == (@__FILE__) && test_externally_allocated_memory()
