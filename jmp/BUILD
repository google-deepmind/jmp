# Description: JMP is a JAX Mixed Precision library.
package(default_visibility = ["//visibility:private"])

licenses(["notice"])

exports_files(["LICENSE"])

# TODO(tomhennigan): This should be public.
package_group(
    name = "friends",
    packages = ["//jmp/..."],
)

py_library(
    name = "jmp",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = [":friends"],
    deps = [
        "//jmp/_src:loss_scale",
        "//jmp/_src:policy",
    ],
)
