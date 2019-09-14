from cx_Freeze import setup, Executable
import os

# Extra files
include_files = []

# Including libs
includes = []

# Unnecessary libs
excludes = []

bin_includes = []
# ,'idna.idnadata'
# Packages
packages = []

# Optimization level
optimize = 0

# If true, only errors and warning will be displayed when excuting cx_freeze
silent = True


target = Executable(
    script = "gmg.py",
    copyright= "Copyright © 2019 GMG",
    )


setup(name = "Guide_Me_Glasses",
      version = "0.1" ,
      description = "" ,
      options = {'build_exe': {'includes':includes,'excludes':excludes,'bin_includes':bin_includes, 'packages':packages, 'include_files':include_files, 'optimize':1, 'silent':silent}},
      executables = [target])