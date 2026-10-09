# Run fuzz.jl in separate processes to isolate crashes and release compiled types between rounds.
# Check copies, copies of copies, Base.deepcopy and wrapper calls across GC.
# Failed checks report family, round and seed.
#
# From the package directory:
#     julia scripts/fuzz_hammer.jl 60                  # launch for 60 minutes
#     julia scripts/fuzz_hammer.jl 60 4                # with 4 workers
#     julia scripts/fuzz_hammer.jl reproduce 7 42
#     julia scripts/fuzz_hammer.jl reproduce 7 method
# Reproduce accepts a seed for all families, or a family to scan until failure.
# A family scan prints each seed before copying. Both forms fail for failed checks.
# Direct runs use Pkg.test in a temporary environment for test dependencies. The driver waits for
# active workers after its launch deadline. Logs go to scripts/logs/hammer.log; failed worker logs stay
# beside it.

# Workers inherit the test environment; other direct invocations create one through Pkg.test.
if abspath(PROGRAM_FILE) == (@__FILE__) && (isempty(ARGS) || ARGS[1] != "worker")
    import Pkg
    Pkg.activate(mktempdir(); io = devnull)
    Pkg.develop(path = dirname(@__DIR__); io = devnull)
    try
        Pkg.test("SpecializedDeepcopy"; test_args = ["hammer", ARGS...])
    catch e
        e isa Pkg.Types.PkgError || rethrow()
        exit(1)
    end
    exit(0)
end

using SpecializedDeepcopy
using Random: Xoshiro
# The test suite includes this file into a module that already imports the suite's Fuzz.
isdefined(@__MODULE__, :Fuzz) || include(joinpath(dirname(@__DIR__), "test", "fuzz.jl"))

const ROUND_SECONDS = 120
const PROGRESS_EVERY = 25

function hammer(args)
    if !isempty(args) && args[1] == "worker"
        return worker(parse(Int, args[2]), parse(Float64, args[3]))
    elseif !isempty(args) && args[1] == "reproduce"
        seed = tryparse(Int, args[3])
        return reproduce(parse(Int, args[2]), seed === nothing ? args[3] : seed)
    end
    return driver(settings(args)...)
end

# The duration in minutes and the number of workers of a run.
function settings(args)
    minutes = isempty(args) ? 60.0 : parse(Float64, args[1])
    workers = length(args) >= 2 ? parse(Int, args[2]) : max(1, Sys.CPU_THREADS ÷ 2)
    # A run without time or workers would report success without copying anything.
    isfinite(minutes) && minutes > 0 || throw(ArgumentError("the duration must be a positive number of minutes, not $minutes"))
    workers > 0 || throw(ArgumentError("the number of workers must be positive, not $workers"))
    return minutes, workers
end

# Plain, custom-method and trim-oriented layouts; this runner does not compile trimmed programs.
function families(round)
    rng = Xoshiro(round)
    return [("plain", Fuzz.define_types!(rng, rand(rng, 6:14)), true),
        ("method", Fuzz.define_types!(rng, rand(rng, 6:14); methods = true), true),
        ("trim", Fuzz.define_types!(rng, rand(rng, 6:14); trim = true, methods = true), true)]
end

# Return failed checks for one generated value.
function exercise(family, with_base, round, seed)
    rng = Xoshiro(hash((round, seed)))
    depth, width = rand(rng, 2:5), rand(rng, 1:5)
    value = Fuzz.generate(rng, family; depth, width)
    reports = String[]
    record(stage, report) = report === nothing || push!(reports, stage * ": " * report)
    copied = specialized_deepcopy(value)
    record("copy", Fuzz.check(value, copied))
    record("copy of the copy", Fuzz.check(copied, specialized_deepcopy(copied)))
    with_base && record("Base.deepcopy", Fuzz.check(value, deepcopy(value); bits_fields = false))
    # Check wrapper calls after GC. Dropping the input reference does not prove it was collected.
    wrappers = Any[o for o in Fuzz.mutables(copied) if o isa Fuzz.FunctionWrapper]
    expected = Any[wrapper(1) for wrapper in wrappers]
    value = nothing
    GC.gc()
    for (wrapper, result) in zip(wrappers, expected)
        wrapper(1) == result || push!(reports, "wrapper: $(wrapper(1)) instead of $result")
    end
    return reports
end

function worker(round, seconds)
    deadline = time() + seconds
    seed = 0
    failures = 0
    for (name, family, with_base) in families(round)
        seed = 0
        println("FAMILY round=$round family=$name")
        flush(stdout)
        while time() < deadline
            seed += 1
            reports = try
                # The generated constructors are newer than this method's world age.
                Base.invokelatest(exercise, family, with_base, round, seed)
            catch e
                ["exception: " * sprint(showerror, e)]
            end
            for report in reports
                failures += 1
                println("FAIL round=$round seed=$seed family=$name $report")
            end
            if seed % PROGRESS_EVERY == 0
                println("PROGRESS round=$round family=$name seed=$seed")
                flush(stdout)
            end
        end
        deadline = time() + seconds  # start the next family's budget
    end
    println("DONE round=$round failures=$failures")
    return failures == 0 ? 0 : 1
end

# Run one seed in all families, or scan a named family until failure. Return 1 for failed checks.
function reproduce(round, target)
    target isa Int || target in ("plain", "method", "trim") || throw(ArgumentError("no family named $target"))
    failed = false
    for (name, family, with_base) in families(round)
        target isa Int || target == name || continue
        for seed in (target isa Int ? (target,) : Iterators.countfrom(1))
            target isa Int || println("seed $seed")
            flush(stdout)
            reports = Base.invokelatest(exercise, family, with_base, round, seed)
            for report in reports
                println("FAIL round=$round seed=$seed family=$name $report")
            end
            failed |= !isempty(reports)
            failed && !(target isa Int) && break
        end
    end
    return failed ? 1 : 0
end

# Keep the driver log and failed worker logs. Remove successful and previous runs' worker logs.
const LOGS = joinpath(@__DIR__, "logs")

# Start `command` with both output streams in the file `log`. One handle serves both: two handles to
# one file write at separate offsets, over each other's output.
launch(command, log) = open(io -> run(pipeline(command; stdout = io, stderr = io); wait = false), log, "w")

# A worker that completes prints DONE and exits with status 1 exactly when it printed a FAIL line.
completed(lines, exitcode, termsignal) =
    any(startswith("DONE"), lines) && termsignal == 0 && exitcode == (any(startswith("FAIL"), lines) ? 1 : 0)

function driver(minutes, workers)
    mkpath(LOGS)
    foreach(name -> rm(joinpath(LOGS, name)), filter(startswith("hammer-round-"), readdir(LOGS)))
    return open(joinpath(LOGS, "hammer.log"), "w") do io
        say(line) = (println(io, line); flush(io); println(line); flush(stdout))
        deadline = time() + 60minutes
        started = time()
        rounds = 0
        seeds = 0
        failures = String[]
        running = Dict{Int, Tuple{Base.Process, String}}()
        report = time()
        say("hammer: $(round(Int, minutes)) minutes with $workers workers, $ROUND_SECONDS s per round and family, logs in $LOGS")
        while time() < deadline || !isempty(running)
            while length(running) < workers && time() < deadline
                rounds += 1
                log = joinpath(LOGS, "hammer-round-$rounds.log")
                command = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) $(@__FILE__) worker $rounds $ROUND_SECONDS`
                running[rounds] = (launch(command, log), log)
            end
            sleep(1)
            for (r, (process, log)) in collect(running)
                process_exited(process) || continue
                delete!(running, r)
                lines = readlines(log)
                fails = filter(startswith("FAIL"), lines)
                for line in fails
                    push!(failures, line)
                    say(line)
                end
                progress = filter(startswith("PROGRESS"), lines)
                for family in ("plain", "method", "trim")
                    last = findlast(l -> occursin("family=$family", l), progress)
                    last === nothing || (seeds += parse(Int, split(progress[last], "seed=")[2]))
                end
                crashed = !completed(lines, process.exitcode, process.termsignal)
                if crashed
                    tail = join(lines[max(1, end - 20):end], "\n")
                    begun = filter(startswith("FAMILY"), lines)
                    family = isempty(begun) ? "plain" : split(last(begun), "family=")[2]
                    push!(failures, "CRASH round=$r family=$family after $(isempty(progress) ? "no progress" : progress[end])")
                    say("CRASH round=$r family=$family, exit code $(process.exitcode): reproduce it with `reproduce $r $family`, worker log $log\n$tail")
                end
                isempty(fails) && !crashed && rm(log; force = true)
            end
            if time() - report > 60
                report = time()
                say("hammer: $(rounds - length(running)) rounds done, about $seeds seeds, $(length(failures)) failures, $(round(Int, (time() - started) / 60)) min")
            end
        end
        say("hammer: $rounds rounds, about $seeds seeds, $(length(failures)) failures")
        isempty(failures) || say("reproduce a failure with `julia scripts/fuzz_hammer.jl reproduce <round> <seed or family>`; the worker logs of failed rounds are in $LOGS")
        return isempty(failures) ? 0 : 1
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(hammer(ARGS))
end
