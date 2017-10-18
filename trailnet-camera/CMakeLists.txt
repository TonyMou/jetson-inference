
file(GLOB trailnetCameraSources *.cpp)
file(GLOB trailnetCameraIncludes *.h )

cuda_add_executable(trailnet-camera ${trailnetCameraSources})
target_link_libraries(trailnet-camera nvcaffe_parser nvinfer jetson-inference)
