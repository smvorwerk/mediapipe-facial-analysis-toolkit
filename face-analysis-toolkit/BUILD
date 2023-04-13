licenses(["notice"])

package(default_visibility = [
    "//visibility:public",
])

cc_library(
    name = "demo",
    srcs = ["demo.cc"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:opencv_highgui",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:opencv_video",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
        "//mp_proctor/calculators/util:proctor_result",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
    ],
)

cc_binary(
    name = "demo_app",
    deps = [
        ":demo",
        "//mp_proctor/graphs/face_mesh/full:live_calculators",
    ],
)
