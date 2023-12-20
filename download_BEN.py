#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Your Script Name Here
Description: A brief description of what your script does.
"""

# Imports
import torchgeo
import sys
import os

from torchgeo.datasets.bigearthnet import BigEarthNet
 

# Functions
def main():
    root = os.getcwd() + '/data'
    BENDatasets = BigEarthNet(root=root)

# Entry point of the script
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

