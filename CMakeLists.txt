cmake_minimum_required(VERSION 3.27)
project(xmotion VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(TFLITE_ENABLE_GPU ON CACHE BOOL "Enable TFLite GPU support")
set(ABSL_PROPAGATE_CXX_STD ON)
set(JSON_BuildTests OFF CACHE INTERNAL "")
set(CL_TARGET_OPENCL_VERSION 300)

add_subdirectory(external/tensorflow/tensorflow/lite)
add_subdirectory(external/argparse)
add_subdirectory(external/json)
add_subdirectory(external/glm)

set(CMAKE_C_COMPILER "clang-16")
set(CMAKE_CXX_COMPILER "clang++-16")

# Creates the compile_commands.json used by editors for code completion
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Must use GNUInstallDirs to install libraries into correct locations on all platforms:
include(GNUInstallDirs)

# Find OpenCL
find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Find OpenCV package
find_package(OpenGL REQUIRED)

# Find spdlog package
find_package(spdlog REQUIRED)

# Find GTK-MM package
find_package(PkgConfig REQUIRED)
pkg_check_modules(GTKMM REQUIRED gtkmm-3.0)

if (UNIX AND NOT APPLE)
    # Find v4l2 (video for linux) package
    pkg_check_modules(V4L2 REQUIRED libv4l2)
endif ()


set(SOURCE_FILES
        main.cpp

        platforms/agnostic_cap.h

        xmotion/utils/velocity_filter.h
        xmotion/utils/low_pass_filter.h
        xmotion/dnn/net/pose_detector.h
        xmotion/dnn/net/dnn_runner.h
        xmotion/dnn/net/blaze_pose.h
        xmotion/dnn/net/dnn_common.h
        xmotion/dnn/net/ssd_anchors.h
        xmotion/dnn/net/pose_roi.h
        xmotion/dnn/net/roi_predictor.h
        xmotion/dnn/pose_pipeline.h
        xmotion/utils/thread_pool.h
        xmotion/utils/timer.h
        xmotion/utils/delta_loop.h
        xmotion/utils/eox_globals.h
        xmotion/boot/i_boot.h
        xmotion/boot/file_boot.h
        xmotion/boot/gtk_boot.h
        xmotion/ogl/texture_1.h
        xmotion/ogl/simple_shader.h
        xmotion/gtk/gl_image.h
        xmotion/gtk/simple_image_window.h
        xmotion/boot/a_updated_boot.h
        xmotion/gtk/small_button.h
        xmotion/data/json_config.h
        xmotion/camera/stereo_camera.h
        xmotion/camera/d_dummy_camera.h
        xmotion/gtk/gtk_utils.h
        xmotion/gtk/gtk_cam_params.h
        xmotion/gtk/cam_params_window.h
        xmotion/gtk/gtk_config_stack.h
        xmotion/algo/i_logic.h
        xmotion/algo/calibration.h
        xmotion/algo/pose.h
        xmotion/utils/cv_utils.h
        xmotion/data/json_ocv.h
        xmotion/algo/cross.h

        sources/velocity_filter.cpp
        sources/low_pass_filter.cpp
        sources/pose_detector.cpp
        sources/blaze_pose.cpp
        sources/dnn_common.cpp
        sources/ssd_anchors.cpp
        sources/pose_roi.cpp
        sources/thread_pool.cpp
        sources/timer.cpp
        sources/delta_loop.cpp
        sources/eox_globals.cpp
        sources/file_boot.cpp
        sources/gtk_boot.cpp
        sources/texture_1.cpp
        sources/simple_shader.cpp
        sources/gl_image.cpp
        sources/simple_image_window.cpp
        sources/small_button.cpp
        sources/json_config.cpp
        sources/stereo_camera.cpp
        sources/a_updated_boot.cpp
        sources/gtk_cam_params.cpp
        sources/cam_params_window.cpp
        sources/gtk_config_stack.cpp
        sources/file_boot_gui.cpp
        sources/file_boot_cam.cpp
        sources/calibration.cpp
        sources/pose.cpp
        sources/file_boot_logic.cpp
        sources/cv_utils.cpp
        sources/json_ocv.cpp
        sources/cross.cpp
        sources/pose_pipeline.cpp
        sources/pose_pipeline_debug.cpp
        sources/pose_pipeline_aux.cpp
        sources/d_dummy_camera.cpp
)

if (UNIX AND NOT APPLE)

    list(APPEND SOURCE_FILES platforms/unix/v4l2/linux_cap.cpp)
    list(APPEND SOURCE_FILES platforms/unix/v4l2/linux_video.h)
    list(APPEND SOURCE_FILES platforms/unix/v4l2/linux_video.cpp)

elseif (WIN32)

    list(APPEND SOURCE_FILES platforms/win32/direct_show/windows_cap.cpp)
    list(APPEND SOURCE_FILES platforms/win32/direct_show/windows_video.h)
    list(APPEND SOURCE_FILES platforms/win32/direct_show/windows_video.cpp)

endif ()


add_executable(xmotion ${SOURCE_FILES})


if (WIN32)

    # Include directories for the specific target
    target_include_directories(${PROJECT_NAME}
            PRIVATE ${OpenCV_INCLUDE_DIRS}
            PRIVATE ${GTKMM_INCLUDE_DIRS})

    # Link libraries
    target_link_libraries(${PROJECT_NAME}
            mfplat mf mfreadwrite
            PRIVATE ${OpenCV_LIBS}
            PRIVATE ${GTKMM_LIBRARIES}
            PRIVATE OpenGL::GL
            PRIVATE spdlog::spdlog
            PRIVATE nlohmann_json::nlohmann_json
            PRIVATE tensorflow-lite
            PRIVATE argparse
            PRIVATE glm)

    target_compile_options(${PROJECT_NAME}
            PRIVATE ${GTKMM_CFLAGS_OTHER})

elseif (UNIX AND NOT APPLE)

    # Include directories for the specific target
    target_include_directories(${PROJECT_NAME}
            PRIVATE ${OpenCV_INCLUDE_DIRS}
            PRIVATE ${GTKMM_INCLUDE_DIRS}
            PRIVATE ${V4L2_INCLUDE_DIRS})

    # Link libraries
    target_link_libraries(${PROJECT_NAME}
            PRIVATE ${OpenCV_LIBS}
            PRIVATE ${GTKMM_LIBRARIES}
            PRIVATE OpenGL::GL
            PRIVATE spdlog::spdlog
            PRIVATE ${V4L2_LIBRARIES}
            PRIVATE nlohmann_json::nlohmann_json
            PRIVATE tensorflow-lite
            PRIVATE argparse
            PRIVATE glm)

    target_compile_options(${PROJECT_NAME}
            PRIVATE ${GTKMM_CFLAGS_OTHER}
            PRIVATE ${V4L2_CFLAGS_OTHER})

endif ()