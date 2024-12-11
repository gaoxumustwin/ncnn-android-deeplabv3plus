project(deeplabv3plusncnn)

cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 11)

set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/opencv-mobile-4.5.1-android/sdk/native/jni)
find_package(OpenCV REQUIRED core imgproc)

set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnn-20240410-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

add_library(deeplabv3plusncnn SHARED deeplabv3plusncnn_jni.cpp)

target_link_libraries(deeplabv3plusncnn ncnn ${OpenCV_LIBS})
