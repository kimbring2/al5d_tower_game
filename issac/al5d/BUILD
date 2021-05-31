load("//bzl:module.bzl", "isaac_app", "isaac_cc_module")
load("@com_nvidia_isaac_sdk//bzl:module.bzl", "isaac_app")

isaac_cc_module(
  name = "al5d_components",
  srcs = ["Al5d.cpp"],
  hdrs = ["Al5d.hpp"],
)

isaac_app(
     name = "al5d",
     modules = [
     	"//packages/al5d:al5d_components"
     ],
)