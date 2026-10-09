# Compile fresh trim layouts with juliac --trim=safe, one generated value per program, and run each
# program. The `trim` family of fuzz_hammer.jl restricts layouts but does not compile them with
# --trim; this script does. Each round builds 3 values of a new trim family, one program after the
# other, all from test/trim/fuzz_probe.jl: the build reads the round and the value from the
# environment.
#
# From the package directory, with Julia 1.12 or later:
#     julia scripts/fuzz_trim.jl 60        # build programs for 60 minutes
#
# juliac builds a program in a separate Julia process, which reads the program from its file and
# writes an object file; the trim verifier runs only in that process. The builds share the directory
# build/ of the run: each build replaces its image, program and log. JuliaC also copies the build
# environment to a temporary directory for each build.
# A run first builds a control program that the verifier must reject, and stops if it is not
# rejected. Each program copies its value and prints whether the copy has the type of the value, so a
# pass shows that the copy compiles and runs under --trim=safe; test/trim/main.jl checks values and
# aliases of selected fixtures.
# The outcome of a program is `passed`, `rejected` (a verifier error that names SpecializedDeepcopy),
# `build failed`, `link failed` or `runtime failed`. The trim family is meant to compile, so every
# outcome other than `passed` is a finding. results.tsv has a line for each program: round, value,
# outcome and seconds. A program that does not pass keeps its build log and output as
# round-<n>-<i>.log. Rounds count from 1 in each run. Each run draws a run seed, and the run seed and
# the round select the layouts, so each run builds new layouts. To reproduce a program, build
# test/trim/fuzz_probe.jl with FUZZ_TRIM_RUN_SEED=<run seed>, FUZZ_TRIM_ROUND=<n> and
# FUZZ_TRIM_VALUE=<i> and the environment of the run. The run directory
# scripts/logs/run-<date>-<time>-trim also holds a copy of the package source, the build environment
# and run.toml, which records the run seed. Ctrl-C ends the run after the summary.

const VALUES_PER_ROUND = 3

if abspath(PROGRAM_FILE) == (@__FILE__) && (isempty(ARGS) || ARGS[1] != "build")
    VERSION >= v"1.12" || error("juliac --trim needs Julia 1.12 or later, not $VERSION")
    import Pkg
    minutes = isempty(ARGS) ? 60.0 : parse(Float64, ARGS[1])
    isfinite(minutes) && minutes > 0 || throw(ArgumentError("the duration must be a positive number of minutes, not $minutes"))
    # Copy the package source, so that the builds use this version.
    package = dirname(@__DIR__)
    run_dir = let stamp = Libc.strftime("%Y%m%d-%H%M%S", time()), n = 1
        candidate = joinpath(@__DIR__, "logs", "run-$stamp-trim")
        while ispath(candidate)
            n += 1
            candidate = joinpath(@__DIR__, "logs", "run-$stamp-$n-trim")
        end
        candidate
    end
    source = mkpath(joinpath(run_dir, "source"))
    foreach(name -> cp(joinpath(package, name), joinpath(source, name)), ("Project.toml", "src", "ext", "test"))
    mkpath(joinpath(source, "scripts"))
    for name in filter(endswith(".jl"), readdir(@__DIR__))
        cp(joinpath(@__DIR__, name), joinpath(source, "scripts", name))
    end
    environment = mkpath(joinpath(run_dir, "environment"))
    uuid = Pkg.TOML.parsefile(joinpath(package, "Project.toml"))["uuid"]
    write(joinpath(environment, "Project.toml"), """
        [deps]
        FunctionWrappers = "069b7b12-0de2-55c6-9aab-29f3d0a68a2e"
        JuliaC = "acedd4c2-ced6-4a15-accc-2607eb759ba2"
        SpecializedDeepcopy = "$uuid"

        [compat]
        FunctionWrappers = "1.1.2"
        JuliaC = "0.3"

        [sources]
        SpecializedDeepcopy = {path = $(repr(source))}
        """)
    Pkg.activate(environment; io = devnull)
    Pkg.instantiate(; io = devnull)
    Base.exit_on_sigint(false)
    build = `$(Base.julia_cmd()) --startup-file=no --project=$environment $(joinpath(source, "scripts", "fuzz_trim.jl")) build $run_dir $minutes`
    process = run(pipeline(build; stdout, stderr); wait = false)
    # Ctrl-C reaches both processes; the build process stops after its current step and writes its
    # summary, so wait for it.
    try
        wait(process)
    catch e
        e isa InterruptException || rethrow()
        wait(process)
    end
    exit(success(process) ? 0 : 1)
end

using JuliaC

# Run `f` with its output in `log`. Return false when it throws, with the error in the log.
function logged(f, log; mode = "a")
    return open(log, mode) do io
        redirect_stdout(io) do
            redirect_stderr(io) do
                try
                    f()
                    true
                catch e
                    e isa InterruptException && rethrow()
                    showerror(io, e)
                    false
                end
            end
        end
    end
end

# Build test/trim/fuzz_probe.jl for the round and value in FUZZ_TRIM_ROUND and FUZZ_TRIM_VALUE.
# Return (outcome, image recipe); a failed compilation has no recipe. As in test/trim/build.jl.
function compile(run_dir, log)
    image = JuliaC.ImageRecipe(; output_type = "--output-exe", trim_mode = "safe",
        file = joinpath(run_dir, "source", "test", "trim", "fuzz_probe.jl"),
        project = joinpath(run_dir, "environment"), img_path = joinpath(run_dir, "build", "image.o.a"))
    logged(() -> JuliaC.compile_products(image), log; mode = "w") && return "compiled", image
    # Strip ANSI color before matching either verifier diagnostic format.
    text = replace(read(log, String), r"\e\[[0-9;]*[A-Za-z]" => "")
    verified = occursin("Verifier error", text) || occursin(r"Trim verify finished with [1-9]\d* errors", text)
    rejected = verified && occursin("SpecializedDeepcopy", text)
    return rejected ? "rejected" : "build failed", nothing
end

# Build, link and run the program of value `i` of round `r`. Return its outcome. A program that does
# not pass keeps its log and output as round-<r>-<i>.log.
function probe(run_dir, run_seed, r, i)
    build = joinpath(run_dir, "build")
    log, output = joinpath(build, "build.log"), joinpath(build, "output.txt")
    rm(output; force = true)
    outcome = withenv("FUZZ_TRIM_RUN_SEED" => string(run_seed), "FUZZ_TRIM_ROUND" => string(r), "FUZZ_TRIM_VALUE" => string(i)) do
        compiled, image = compile(run_dir, log)
        compiled == "compiled" || return compiled
        link = JuliaC.LinkRecipe(; image_recipe = image, outname = joinpath(build, "probe"))
        logged(() -> JuliaC.link_products(link), log) || return "link failed"
        # The linked program loads the libraries of this Julia installation, so it runs in place. A
        # bundle copies about 120 MiB of libraries, so only a program that fails in place gets one,
        # for a system where the program does not find those libraries.
        exe = Sys.iswindows() ? ".exe" : ""
        passes(program) = open(io -> success(pipeline(ignorestatus(`$program`); stdout = io, stderr = io)), output, "w") &&
            strip(read(output, String)) == "true"
        passes(joinpath(build, "probe" * exe)) && return "passed"
        logged(() -> JuliaC.bundle_products(JuliaC.BundleRecipe(; link_recipe = link, output_dir = build)), log) || return "link failed"
        return passes(joinpath(build, "bin", "probe" * exe)) ? "passed" : "runtime failed"
    end
    if outcome != "passed"
        kept = joinpath(run_dir, "round-$r-$i.log")
        cp(log, kept; force = true)
        isfile(output) && open(io -> write(io, "\noutput:\n", read(output)), kept, "a")
    end
    return outcome
end

# The control program has an abstract element type, which needs runtime dispatch. The verifier must
# reject it; otherwise the builds do not check trim safety, and a pass means nothing.
function control(run_dir)
    log = joinpath(run_dir, "build", "control.log")
    outcome, _ = withenv(() -> compile(run_dir, log), "FUZZ_TRIM_ROUND" => "control")
    outcome == "rejected" || cp(log, joinpath(run_dir, "control.log"); force = true)
    return outcome
end

function lane(run_dir, minutes)
    deadline = time() + 60minutes
    run_seed = rand(UInt32)
    next = 1
    outcomes = Dict{String, Int}()
    seconds = Float64[]
    say(line) = (open(io -> println(io, line), joinpath(run_dir, "trim.log"), "a"); println(line); flush(stdout))
    function settings(status)
        open(joinpath(run_dir, "run.toml"), "w") do io
            println(io, "mode = \"trim\"\njulia = \"$VERSION\"\nmachine = \"$(Sys.MACHINE)\"\nword_size = $(Sys.WORD_SIZE)")
            println(io, "run_seed = $run_seed\nrounds = $(next - 1)\nprograms = $(length(seconds))\nstatus = \"$status\"")
        end
    end
    settings("running")
    say("trim: $(round(Int, minutes)) minutes, run seed $run_seed, $VALUES_PER_ROUND programs per round, run directory $run_dir")
    mkpath(joinpath(run_dir, "build"))
    interrupted = false
    try
        checked = control(run_dir)
        if checked != "rejected"
            say("trim: the control program was $checked, not rejected, so the builds do not check trim safety; see $(joinpath(run_dir, "control.log"))")
            settings("failed")
            return 1
        end
        while time() < deadline
            r = next
            next += 1
            for i in 1:VALUES_PER_ROUND
                start = time()
                outcome = probe(run_dir, run_seed, r, i)
                push!(seconds, time() - start)
                outcomes[outcome] = get(outcomes, outcome, 0) + 1
                open(io -> println(io, r, '\t', i, '\t', outcome, '\t', round(seconds[end]; digits = 1)), joinpath(run_dir, "results.tsv"), "a")
                outcome == "passed" || say("trim: round $r value $i $outcome; see $(joinpath(run_dir, "round-$r-$i.log"))")
            end
        end
    catch e
        e isa InterruptException || rethrow()
        interrupted = true
    end
    rm(joinpath(run_dir, "build"); recursive = true, force = true)
    built = length(seconds)
    mean = built == 0 ? 0.0 : round(sum(seconds) / built; digits = 1)
    say("trim: $(interrupted ? "interrupted, " : "")$(next - 1) rounds, $built programs, $mean s per program; " *
        join(("$n $k" for (k, n) in sort!(collect(outcomes); by = first)), ", "))
    passed = !interrupted && built > 0 && get(outcomes, "passed", 0) == built
    settings(interrupted ? "interrupted" : passed ? "passed" : "failed")
    return passed ? 0 : 1
end

if abspath(PROGRAM_FILE) == @__FILE__
    # A script exits at once on Ctrl-C; with that off, the lane stops after the current step and
    # writes its summary.
    Base.exit_on_sigint(false)
    exit(lane(ARGS[2], parse(Float64, ARGS[3])))
end
