licenses(["notice"])

package(default_visibility = ["//visibility:private"])

cc_library(name = "face_orientation_calculator",
    srcs        = ["face_orientation_calculator.cc"],
    visibility  = ["//visibility:public"],
    deps        = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/util:render_data_cc_proto",
    ],
    alwayslink = 1,
)

cc_library(name = "face_orientation_to_render_data_calculator",
    srcs        = ["face_orientation_to_render_data_calculator.cc"],
    visibility  = ["//visibility:public"],
    deps        = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/util:render_data_cc_proto",
    ],
    alwayslink = 1,
)

