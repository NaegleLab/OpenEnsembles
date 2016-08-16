

""" This is test folder for running unit tests.
"""
import sys
import os
# Add the parent directory (which holds the localcider package)
print(__file__)
sys.path.insert(0, os.path.abspath(__file__ + "/../../"))

#from . import unit_tests
import unit_tests


#import test_sequenceParameters
#import test_plots
