licenses(["notice"])

package(default_visibility = ["//visibility:public"])

exports_files(
    srcs = glob(
        ["*.pbtxt"]
    )
)

cc_library(
    name = "live_calculators",
    deps = [
        "//mp_proctor/calculators/util:proctor_result_calculator",
        "//mp_proctor/calculators/util:proctor_result",
        "//mediapipe/calculators/core:flow_limiter_calculator",
        "//mp_proctor/graphs/face_mesh:facemesh_deps",
    ],
)
