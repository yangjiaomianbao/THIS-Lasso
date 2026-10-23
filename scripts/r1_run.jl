# R1 command logger. Outputs remain inside results/r1, within this stage's boundary.
using Dates
root = normpath(joinpath(@__DIR__, ".."))
length(ARGS) in (1, 2) || error("Usage: r1_run.jl TASK [unique_log_label]")
task = ARGS[1]
targets = Dict("test" => "tests/test_r1_baseline.jl", "baseline" => "scripts/r1_baseline.jl", "replay" => "scripts/r1_replay.jl",
    "revision_test" => "tests/test_r1_revision.jl",
    "revision" => "scripts/r1_revision.jl", "revision_replay" => "scripts/r1_revision_replay.jl")
haskey(targets, task) || error("Unknown R1 task")
label = length(ARGS) == 2 ? ARGS[2] : task
occursin(r"^[A-Za-z0-9_-]+$", label) || error("Invalid log label")
log_dir = startswith(task, "revision") ? joinpath(root, "results", "r1", "revision", "logs") :
    joinpath(root, "results", "r1", "logs")
mkpath(log_dir)
log_path = joinpath(log_dir, label * ".log")
ispath(log_path) && error("Log exists; choose a new label to retain previous evidence")
wrapper = joinpath(root, "environment", "r0", "julia.sh")
project = joinpath(root, "environment", "r0", "julia")
target = joinpath(root, targets[task])
cmd = `$wrapper --startup-file=no --project=$project $target`
code = 1
open(log_path, "w") do io
    println(io, "START=", now(UTC) + Hour(8), "+08:00")
    println(io, "CWD=", root, "\nCOMMAND=", cmd)
    println(io, "ENV=project wrapper sets independent Julia depot and OPENBLAS_NUM_THREADS=1")
    flush(io)
    process = run(pipeline(ignorestatus(Cmd(cmd; dir=root)), stdout=io, stderr=io))
    global code = process.exitcode
    println(io, "EXIT_CODE=", code)
    println(io, "END=", now(UTC) + Hour(8), "+08:00")
end
print(read(log_path, String))
println("LOG=", log_path)
exit(code)
