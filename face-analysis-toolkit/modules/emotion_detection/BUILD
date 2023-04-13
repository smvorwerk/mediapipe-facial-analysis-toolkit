load(
    "//mediapipe/framework/tool:mediapipe_graph.bzl",
    "mediapipe_simple_subgraph",
)

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

exports_files([
	"emotion_detection.tflite"
])

mediapipe_simple_subgraph(
    name = "face_emotion_cpu",
    graph = "face_emotion_cpu.pbtxt",
    register_as = "FaceEmotionCpu",
    deps = [
		"//mediapipe/calculators/image:image_cropping_calculator",
		"//mediapipe/calculators/core:gate_calculator",
        "//mediapipe/calculators/core:split_vector_calculator",
        "//mediapipe/calculators/tensor:image_to_tensor_calculator",
        "//mediapipe/calculators/tensor:inference_calculator",
        "//mediapipe/calculators/tensor:tensors_to_floats_calculator",
        "//mediapipe/calculators/tensor:tensors_to_landmarks_calculator",
		"//mediapipe/calculators/tensor:tensors_to_classification_calculator",
		"//mediapipe/calculators/tflite:tflite_tensors_to_classification_calculator",
		"//mediapipe/calculators/tflite:tflite_custom_op_resolver_calculator",
        "//mediapipe/calculators/tflite:tflite_inference_calculator",
		"//mediapipe/calculators/tflite:tflite_converter_calculator",
		"//mediapipe/calculators/util:landmark_projection_calculator",
        "//mediapipe/calculators/util:thresholding_calculator",
		"//mediapipe/calculators/util:detection_label_id_to_text_calculator",
        "//mediapipe/framework/tool:switch_container",
    ],
)