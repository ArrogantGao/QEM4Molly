import cProfile
import pstats
import os

def do_cprofile(filename):
   def wrapper(func):
       def profiled_func(*args, **kwargs):
           # Flag for do profiling or not.*
           DO_PROF = os.getenv("PROFILING")
           if 1:
               profile = cProfile.Profile()
               profile.enable()
               result = func(*args, **kwargs)
               profile.disable()
               # Sort stat by internal time.*
               sortby = "tottime"
               ps = pstats.Stats(profile).sort_stats(sortby)
               ps.dump_stats(filename)
           else:
               result = func(*args, **kwargs)
           return result
       return profiled_func
   return wrapper