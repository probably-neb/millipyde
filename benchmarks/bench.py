from typing import List, Tuple, Callable
import logging
from benchers import millipyde_bench, scikit_image_bench

def get_bench_funcs(bench) -> List[Tuple[ str, Callable ]]:
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

    for bencher in [millipyde_bench, scikit_image_bench]:
        bench = bencher.Bench
        benchmarks = get_bench_funcs(bench)
        from skimage import io
        img = io.imread("benchmarks/inputs/charlie10.png")
        print(bench.name.upper())
        print('=' * 10)
        for name, fn in benchmarks:
            print(name,':',fn(img, "charlie10"))

if __name__ == "__main__":
    main()
