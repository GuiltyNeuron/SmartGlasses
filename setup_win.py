from cx_Freeze import setup, Executable
import os

# Extra files
include_files = []

# Including libs
includes = []

# Unnecessary libs
excludes = []
# Packages
packages = []

# Optimization level
optimize = 1

# If true, only errors and warning will be displayed when excuting cx_freez
silent = True


PYTHON_INSTALL_DIR = os.path.dirname(os.path.dirname(os.__file__))
os.environ['TCL_LIBRARY'] = os.path.join(PYTHON_INSTALL_DIR, 'tcl', 'tcl8.6')
os.environ['TK_LIBRARY'] = os.path.join(PYTHON_INSTALL_DIR, 'tcl', 'tk8.6')


target = Executable(
    script = "gmg.py",
    copyright= "Copyright © 2019 gmg",
    )


setup(name = "Guide_Me_Glasses",
      version = "0.1" ,
      description = "Smart glasses" ,
      options = {'build_exe': {'includes':includes,'excludes':excludes,'packages':packages,'include_files':include_files, 'optimize':1, 'silent':silent}},
      executables = [target])