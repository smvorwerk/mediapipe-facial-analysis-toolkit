licenses(["notice"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "custom_calculators",
    deps = [
        "//mp_proctor/calculators/util:landmark_standardization",
        "//mp_proctor/calculators/face_orientation:face_orientation_calculator",
        "//mp_proctor/calculators/eye_blink:eye_blink_calculator",
        "//mp_proctor/calculators/face_activity:face_movement_calculator",
        "//mp_proctor/calculators/face_activity:face_activity_calculator",
        "//mp_proctor/calculators/util:proctor_result_calculator",
        "//mp_proctor/calculators/util:proctor_result",
    ],
)

cc_library(name = "facemesh_deps",
    deps = [
        "//mediapipe/calculators/core:constant_side_packet_calculator",
        "//mediapipe/calculators/core:flow_limiter_calculator",
        "//mediapipe/calculators/core:packet_cloner_calculator",
        ":custom_calculators"
    ] + select({
        "//mediapipe/gpu:disable_gpu": [
            "//mediapipe/modules/face_landmark:face_landmark_front_cpu",
        ],
        "//conditions:default": [
            "//mediapipe/modules/face_landmark:face_landmark_front_gpu",
        ],
    }),
)

filegroup(name = "minimal_face_mesh_pbtxts",
    srcs = select({
        "//mediapipe/gpu:disable_gpu": [
            "//mp_proctor/graphs/face_mesh/minimal:face_mesh_cpu.pbtxt",
        ],
        "//conditions:default": [
            "//mp_proctor/graphs/face_mesh/minimal:face_mesh_gpu.pbtxt",
        ],
    }),
)

filegroup(name = "face_mesh_data",
    srcs = [
        "//mediapipe/modules/face_detection:face_detection_short_range.tflite",
        "//mediapipe/modules/face_landmark:face_landmark_with_attention.tflite",
    ],
)
