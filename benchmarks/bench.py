from typing import List, Tuple, Callable
import logging


def get_bench_funcs(bench) -> List[Tuple[str, Callable]]:
    funcs = []
    attrs = bench.__dict__
    for name, val in attrs.items():
        if not name.startswith("__") and not name == "name":
            name_fn = name, val
            funcs.append(name_fn)
    return funcs


def main():
    logger = logging.getLogger("benchmarks")
    logger.setLevel(logging.DEBUG)
    data = {}

    import benchers
    for bencher in benchers.load_benchers():
        bench = bencher.Bench
        benchmarks = get_bench_funcs(bench)
        from skimage import io
        img = io.imread("benchmarks/inputs/charlie10.png")
        for name, fn in benchmarks:
            t = fn(img, "charlie10")
            if not data.get(name):
                data[name] = {}
            if not data[name].get(bench.name):
                data[name][bench.name] = {}
            data[name][bench.name]["charlie10"] = t
        from pprint import pprint
    pprint(data)


if __name__ == "__main__":
    main()
