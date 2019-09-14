""" Executable.

Links:

Author: Achraf KHAZRI
Project: GMG
"""

import sys
import argparse
import os
import glob

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--mode', '-mode', help="Classifier mode : task or train", type= str)


# Get args
args = parser.parse_args()

if args.mode == "task":

    print("Hello GMG !")