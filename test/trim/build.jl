# Build --trim=safe tests under ARGS[1], using the environment made by runtests.jl.
# `program/` contains the executable; `shapes/<name>/` contains unlinked probes.
# Linking is unnecessary for the probes: compilation runs the trim verifier.
# `shapes/results.txt` records each name and `compiled`, `rejected` or `failed`.
# Rejection requires a verifier error and SpecializedDeepcopy in the log.
import Pkg
Pkg.instantiate()

using JuliaC

const PROJECT = dirname(Base.active_project())

# Return (outcome, recipe); failed compilations have no recipe.
function compile(file, log)
    image = JuliaC.ImageRecipe(; output_type = "--output-exe", trim_mode = "safe", file, project = PROJECT)
    compiled = open(log, "w") do io
        redirect_stdout(io) do
            redirect_stderr(io) do
                try
                    JuliaC.compile_products(image)
                    true
                catch e
                    showerror(io, e)
                    false
                end
            end
        end
    end
    compiled && return "compiled", image
    # Strip ANSI color before matching either verifier diagnostic format.
    text = replace(read(log, String), r"\e\[[0-9;]*[A-Za-z]" => "")
    verified = occursin("Verifier error", text) || occursin(r"Trim verify finished with [1-9]\d* errors", text)
    rejected = verified && occursin("SpecializedDeepcopy", text)
    return rejected ? "rejected" : "failed", nothing
end

out = only(ARGS)

program = mkpath(joinpath(out, "program"))
_, image = compile(joinpath(@__DIR__, "main.jl"), joinpath(program, "build.log"))
if image !== nothing
    link = JuliaC.LinkRecipe(; image_recipe = image, outname = joinpath(program, "specialized_deepcopy_trim"))
    JuliaC.link_products(link)
    JuliaC.bundle_products(JuliaC.BundleRecipe(; link_recipe = link, output_dir = program))
end

shapes = mkpath(joinpath(out, "shapes"))
names = [m[1] for m in (match(r"^((?:failing_)?shape_\w+)\(\)", line) for line in eachline(joinpath(@__DIR__, "shapes.jl"))) if m !== nothing]
open(joinpath(shapes, "results.txt"), "w") do results
    for name in names
        dir = mkpath(joinpath(shapes, name))
        file = joinpath(dir, "main.jl")
        write(file, """
            include($(repr(joinpath(dirname(@__DIR__), "types.jl"))))
            include($(repr(joinpath(@__DIR__, "shapes.jl"))))
            function @main(ARGS)
                value = $name()
                copied = specialized_deepcopy(value)
                println(Core.stdout, typeof(copied) === typeof(value))
                return 0
            end
            """)
        outcome, _ = compile(file, joinpath(dir, "build.log"))
        println(results, name, " ", outcome)
        flush(results)
    end
end
