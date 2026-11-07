from . import coding_solver, data_solver, generic, math_solver, search_solver


def pick(task):
    t = task.task_type
    instr = task.instruction.lower()
    if t == "coding" or "test_" in instr and "fix" in instr:
        return coding_solver
    if t == "search" or "kb/" in instr:
        return search_solver
    if t == "data_analysis" or ".csv" in instr:
        return data_solver
    if t == "math" or "how many boxes" in instr:
        return math_solver
    return generic
