from smop_functions import *
import smop_functions as sf
from inspect import getmembers, isfunction

global FREQ
global FM_PM
global DURATION_PM
global SAMPLE_RATE

functions_list = [o[0] for o in getmembers(sf) if isfunction(o[1])]
# print(functions_list)

