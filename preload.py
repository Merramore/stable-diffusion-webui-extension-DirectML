#!python3
#preload.py

import inspect, os.path

filename = inspect.getframeinfo(inspect.currentframe()).filename
basename = os.path.split(os.path.dirname(os.path.abspath(filename)))[1]
__package__ = "extensions." + basename
print(f"[DirectML] Extension loaded as {__package__!r}.")


from . import hacks


def preload (parser):
    parser.add_argument("--device", type=str, help="device on which to run. If not specified, try to make an optimal choice.", default=None)
