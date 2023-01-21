import os
import importlib

from .bencher_interface import Benchmarker

"""
Loads all modules in benchers dir. This is required
so that each module is in scope for the Benchmarker.__subclasses__() call
below to work correctly
"""
for module_fs_path in os.listdir(os.path.dirname(__file__)):
    # if "_bench" in module_fs_path:
    #      continue
    module_path = __package__ + '.' + module_fs_path.replace(".py", "")
    importlib.import_module(module_path)


def get_tools():
    """
    Returns every class in benchers dir that is a subclass
    """
    return Benchmarker.__subclasses__()
