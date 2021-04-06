"""Build rules for JMP."""

# NOTE: Internally we swap these out for macros testing on various
#       HW platforms.
jmp_py_binary = native.py_binary
jmp_py_test = native.py_test
jmp_py_library = native.py_library
