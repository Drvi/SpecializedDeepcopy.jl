# The program that scripts/fuzz_trim.jl builds with juliac --trim=safe for each generated value. The
# build reads the run seed from FUZZ_TRIM_RUN_SEED, the round from FUZZ_TRIM_ROUND and the value from
# FUZZ_TRIM_VALUE, so this one file serves every program. The run seed and the round select a trim
# family of 8 types. With FUZZ_TRIM_ROUND=control, the value has an abstract element type, which the
# verifier must reject.
using SpecializedDeepcopy
using Random: Xoshiro
include(joinpath(@__DIR__, "..", "fuzz.jl"))

# Separate statements: the generator calls the constructors of the types that the line before defines.
const CONTROL = ENV["FUZZ_TRIM_ROUND"] == "control"
const LAYOUT = CONTROL ? UInt(0) : hash((parse(UInt, ENV["FUZZ_TRIM_RUN_SEED"]), parse(Int, ENV["FUZZ_TRIM_ROUND"])))
const FAMILY = CONTROL ? DataType[] : Fuzz.define_types!(Xoshiro(LAYOUT), 8; trim = true, methods = true)
const VALUE = CONTROL ? Any[1, "a", [2.0]] : Fuzz.generate(Xoshiro(hash((LAYOUT, parse(Int, ENV["FUZZ_TRIM_VALUE"])))), FAMILY)

function @main(ARGS)
    copied = specialized_deepcopy(VALUE)
    println(Core.stdout, typeof(copied) === typeof(VALUE))
    return 0
end
