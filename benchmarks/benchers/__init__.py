# from .millipyde_bench import *
# from .scikit_image_bench import *
# from .pillow_bench import *
import os
import glob
import importlib


def load_benchers():
    # TODO: replace endswith("_bench.py")
    bencher_fs_paths = list(
        filter(lambda p: p.endswith("_bench.py"),
               os.listdir(os.path.dirname(__file__)))
    )
    bencher_mod_paths = [__package__ + '.' + p.replace(".py", "")
                         for p in bencher_fs_paths]
    print(bencher_mod_paths)
    mods = [importlib.import_module(mod_path)
            for mod_path in bencher_mod_paths]
    return mods
