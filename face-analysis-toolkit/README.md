# MediaPipe Face Analysis Toolkit

The toolkit provides a set of calculators that can be used to analyze the detected faces and their corresponding facial features. The toolkit also provides a set of graphs that can be used to run the calculators and visualize the results in cross-platform applications. This directory represents the customizations to Mediapipe's Core. There is still much to be done!

## Features

- Face Detection and Facial Landmark Detection via [Face Mesh](https://google.github.io/mediapipe/solutions/face_mesh)
- Eye Blink Detection
- Face Orientation (2 DoF, horizontal and vertical)
- Facial Activity
- Face Movement

## TODO
- Face Mesh to Face Landmarker migration
- ARKit52 Facial Action Units blendshape integration
- Improve Gaze Estimation and tracking
- Add export facemesh to *.obj (already created, just not integrated)
- React Native Turbo Modules and Fabric Components
- iOS Integration for Tasks API
- much more...

## Requirement
- Mediapipe v0.8.10.2 (Simply checkout on [this commit](https://github.com/google/mediapipe/commit/63e679d9))

## Installation
To install the toolkit, you need to first install mediapipe and the checkout to the specific version, mentioned previously.
```bash
git clone -n https://github.com/google/mediapipe.git
cd mediapipe
git checkout 63e679d9
```
Then, you can clone this repository under mediapipe root directory.
```sh
git clone https://github.com/sawthiha/mp_proctor.git
```

## Demo Application
To run the demo app, you have to build it first:
```sh
bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mp_proctor:demo_app
```
If there is no build error, you can run the application using the following command.
```sh
GLOG_logtostderr=1 bazel-bin/mp_proctor/demo_app \                     
  --calculator_graph_config_file=mp_proctor/graphs/face_mesh/full/face_mesh_desktop_live.pbtxt
```

## Troubleshooting

### Build errors
#### `fatal error: 'opencv2/core/version.hpp' file not found`
This error occurs when OpenCV installation or config is not detected. If you are on Linux, you can solve this by running the `setup_opencv.sh` provided by Mediapipe. You can find it in the root directory of MediaPipe.
```sh
chmod +x setup_opencv.sh
./setup_opencv.sh
```
