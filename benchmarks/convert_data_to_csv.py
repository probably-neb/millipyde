#!/usr/bin/env python3

import json
import utils
import numpy as np

DATA_DIR = utils.BENCHMARKS_DIR / "data"

AMD_DATA = DATA_DIR / "amd.json"
NVIDIA_DATA = DATA_DIR / "nvidia.json"

AMD_CSV = DATA_DIR / "amd.csv"
NVIDIA_CSV = DATA_DIR / "nvidia.csv"

CSV_ROW = "{tool_name},{rounds},{warmup_rounds},{input_size},{func_name},{min},{max},{mean},{stddev},{median},{iqr},{q1},{q3},{iqr_outliers},{stddev_outliers},{ld15iqr},{hd15iqr},{gc_disabled},{timer}"

CSV_HEADER = CSV_ROW.format(
    tool_name="tool_name",
    rounds="rounds",
    warmup_rounds="warmup_rounds",
    input_size="input_size",
    func_name="func_name",
    min="min",
    max="max",
    mean="mean",
    stddev="stddev",
    median="median",
    iqr="iqr",
    q1="q1",
    q3="q3",
    iqr_outliers="iqr_outliers",
    stddev_outliers="stddev_outliers",
    ld15iqr="ld15iqr",
    hd15iqr="hd15iqr",
    gc_disabled="gc_disabled",
    timer="timer",
)


class BenchmarkTool:
    def __init__(self, name):
        self.name = name
        self.funcs = {k: {s: False for s in [500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500]} for k in utils.benchmarks_list()}
    def set(self, func, size):
        self.funcs[func][size] = True
    def is_complete(self):
        for func in self.funcs:
            for size in self.funcs[func]:
                if not self.funcs[func][size]:
                    return False
        return True


def format_benchmark_csv(benchmark: dict) -> str:
    def f(float_):
        """converts floats to strings without scientific notation"""
        return np.format_float_positional(float_, trim="-")

    return CSV_ROW.format(
        tool_name=benchmark["group"],
        rounds=benchmark["params"]["rounds"],
        warmup_rounds=benchmark["params"]["warmup_rounds"],
        input_size=benchmark["params"]["input_size"],
        func_name=benchmark["extra_info"]["name"],
        min=f(benchmark["stats"]["min"]),
        max=f(benchmark["stats"]["max"]),
        mean=f(benchmark["stats"]["mean"]),
        stddev=f(benchmark["stats"]["stddev"]),
        median=f(benchmark["stats"]["median"]),
        iqr=f(benchmark["stats"]["iqr"]),
        q1=f(benchmark["stats"]["q1"]),
        q3=f(benchmark["stats"]["q3"]),
        iqr_outliers=benchmark["stats"]["iqr_outliers"],
        stddev_outliers=benchmark["stats"]["stddev_outliers"],
        ld15iqr=f(benchmark["stats"]["ld15iqr"]),
        hd15iqr=f(benchmark["stats"]["hd15iqr"]),
        gc_disabled=benchmark["options"]["disable_gc"],
        timer=benchmark["options"]["timer"],
    )
    # ignored fields
    # rounds = benchmark["stats"]["rounds"]
    # outliers = benchmark["stats"]["outliers"]
    # ops = benchmark["stats"]["ops"]
    # total = benchmark["stats"]["total"]
    # iterations = benchmark["stats"]["iterations"]

def create_data_csv(benchmarks) -> str:
    rows = [CSV_HEADER]
    tools = {}

    for benchmark in benchmarks:

        tool_name = benchmark["group"]
        func = benchmark["extra_info"]["name"]
        size = benchmark["params"]["input_size"]
        if tool_name not in tools:
            tools[tool_name] = BenchmarkTool(tool_name)
        tools[tool_name].set(func, size)

        rows.append(format_benchmark_csv(benchmark))
    for tool in tools.values():
        assert tool.is_complete(), f"tool {tool.name} is not complete: {tool}"
    return "\n".join(rows)


AMD_JSON = json.load(open(AMD_DATA, "r"))
AMD_CSV.write_text(create_data_csv(AMD_JSON["benchmarks"]))

NVIDIA_JSON = json.load(open(NVIDIA_DATA, "r"))
NVIDIA_CSV.write_text(create_data_csv(NVIDIA_JSON["benchmarks"]))

AMD_REPORT = DATA_DIR / "amd_info.json"
NVIDIA_REPORT = DATA_DIR / "nvidia_info.json"
if not AMD_REPORT.exists():
    del AMD_JSON["benchmarks"]
    json.dump(AMD_JSON, open(AMD_REPORT, 'w'), indent=2)
else:
    print("not overwriting", AMD_REPORT)
if not NVIDIA_REPORT.exists():
    del NVIDIA_JSON["benchmarks"]
    json.dump(NVIDIA_JSON, open(AMD_REPORT, 'w'), indent=2)
else:
    print("not overwriting",NVIDIA_REPORT)
