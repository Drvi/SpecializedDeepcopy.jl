# A key whose hash reads one of its fields, in a table that the key refers to. The copy reaches the
# table through the key, so the copied table receives the key before the copy sets the key's fields,
# and the key's hash changes after the insertion. specialized_deepcopy checks each table that it
# rebuilt after the copy has set every field, and reinserts the entries of a table that does not
# find one of its keys. Base.deepcopy reads the unset fields of the key instead.
#
# Run this file alone from the package directory: julia --project test/examples/cyclic_key.jl
# The test suite includes this file and calls test_cyclic_key(). The trimmed test program includes
# it for its definitions only.
using SpecializedDeepcopy
using Test

mutable struct CyclicKey
    table::Dict{CyclicKey, Int}
    owner::Vector{Int}
end
Base.hash(k::CyclicKey, h::UInt) = hash(objectid(k.owner), h)

function test_cyclic_key()
    @testset "cyclic key" begin
        key = CyclicKey(Dict{CyclicKey, Int}(), [1])
        key.table[key] = 1
        # Whichever object the copy reaches first, the copied table finds the copied key, and the
        # key refers to the copy of the table.
        for value in ((key,), (key, key.table), (key.table, key), (key.owner, key))
            copied = specialized_deepcopy(value)
            k = only(c for c in copied if c isa CyclicKey)
            @test k.owner == [1] && k.owner !== key.owner && only(keys(k.table)) === k && k.table[k] == 1
            @test all(c -> !(c isa Dict) || c === k.table, copied)
        end

        # Base.deepcopy fails: it crashes the process, raises an exception or makes a table that does
        # not find the key. The child prints "started" first, so a failure before the copy does not
        # count. A working copy means that Base.deepcopy changed, and the README needs an update.
        code = """
            mutable struct CyclicKey
                table::Dict{CyclicKey, Int}
                owner::Vector{Int}
            end
            Base.hash(k::CyclicKey, h::UInt) = hash(objectid(k.owner), h)
            key = CyclicKey(Dict{CyclicKey, Int}(), [1])
            key.table[key] = 1
            print("started ")
            flush(stdout)
            try
                copied = deepcopy(key)
                print("completed: found ", haskey(copied.table, copied))
            catch e
                print(nameof(typeof(e)))
            end
            """
        output, errors = IOBuffer(), IOBuffer()
        success(pipeline(`$(Base.julia_cmd()) --startup-file=no -e $code`; stdout = output, stderr = errors))
        result = String(take!(output))
        base_failed = startswith(result, "started ") && result != "started completed: found true"
        base_failed || print(result, "\n", String(take!(errors)))
        @test base_failed
    end
end

abspath(PROGRAM_FILE) == (@__FILE__) && test_cyclic_key()
