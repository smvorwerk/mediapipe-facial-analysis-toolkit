// Calculator to center align the detected face
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mp_proctor/calculators/util/face_align_calculator.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"

namespace mediapipe {

constexpr char kInputImageTag[] = "IMAGE";
constexpr char kLandmarkTag[] = "LANDMARKS";

// The output side packet containing the cropped image.
constexpr char kOutputTag[] = "ALIGNED";

/**
 * @brief Center Align **a** face in the image given the landmarks and th
 * 
 * INPUTS:
 *      IMAGE - Input Image
 *      LANDMARKS - LandmarksList (Not normlized)
 * 
 * OUTPUTS:
 *      ALIGNED - Aligned Image
 * 
 * Example:
 * 
 * # Face Align Calculator
 *  node  {
 *      calculator: "FaceAlignCalculator"
 *      input_stream: "LANDMARKS:landmarks"
 *      output_stream: "ALIGNED:aligned_image"
 *      node_options: {
 *          [type.googleapis.com/mediapipe.FaceAlignCalculatorOptions] {
 *              interocular_distance: 18
 *              width: -1
 *              height: -1
 *          }
 *      }
 *  }
 * 
 */
class FaceAlignCalculator : public CalculatorBase {
private:
    FaceAlignCalculatorOptions m_options;
public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Tag("IMAGE").Set<ImageFrame>();
    cc->Inputs().Tag("LANDMARKS").Set<LandmarkList>();
    cc->Outputs().Tag(kOutputTag).Set<ImageFrame>();
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Open(CalculatorContext* cc) override {
    // Get the provided options.
    m_options = cc->Options<BlankImageCalculatorOptions>();
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) override {
    // Get the input image.
    const auto& input_image = cc->Inputs().Tag("IMAGE").Get<ImageFrame>();
    cv::Mat image = mediapipe::formats::MatView(&input_image);

    // Create a cv::Rect object representing the crop region.
    cv::Rect crop_region(state_.x - state_.w / 2, state_.y - state_.h / 2,
                         state_.w, state_.h);

    // Crop the image using the cv::Mat::operator() function.
    cv::Mat cropped_image = image(crop_region);

    // Create the output image.
    auto output_image = absl::make_unique<ImageFrame>(
        ImageFormat::SRGB, cropped_image.cols, cropped_image.rows);
    cv::Mat output_image_mat = mediapipe::formats::MatView(output_image.get());

    // Copy the cropped image data into the output image.
    cropped_image.copyTo(output_image_mat);

    // Add the output image to the calculator context.
    cc->Outputs().Tag(kOutputTag).Add(output_image.release(), cc->InputTim
