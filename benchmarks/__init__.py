import sys
import pathlib

cur_lib_path = pathlib.Path().absolute()
sys.path.append(str(cur_lib_path))
import millipyde as mp

INPUTS_PATH = str(cur_lib_path.joinpath("benchmarks/inputs"))
OUTPUTS_PATH = str(cur_lib_path.joinpath("benchmarks/outputs"))
