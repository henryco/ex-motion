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

        xmotion/core/utils/velocity_filter.h
        xmotion/core/utils/low_pass_filter.h
        xmotion/core/dnn/net/pose_detector.h
        xmotion/core/dnn/net/dnn_runner.h
        xmotion/core/dnn/net/blaze_pose.h
        xmotion/core/dnn/net/dnn_common.h
        xmotion/core/dnn/net/ssd_anchors.h
        xmotion/core/dnn/net/pose_roi.h
        xmotion/core/dnn/net/roi_predictor.h
        xmotion/core/dnn/pose_pipeline.h
        xmotion/core/utils/thread_pool.h
        xmotion/core/utils/timer.h
        xmotion/core/utils/delta_loop.h
        xmotion/core/utils/eox_globals.h
        xmotion/core/boot/i_boot.h
        xmotion/fbgtk/file_boot.h
        xmotion/imgui/imgui_boot.h
        xmotion/core/ogl/texture_1.h
        xmotion/core/ogl/simple_shader.h
        xmotion/fbgtk/gtk/gl_image.h
        xmotion/fbgtk/gtk/simple_image_window.h
        xmotion/core/boot/a_updated_boot.h
        xmotion/fbgtk/gtk/small_button.h
        xmotion/fbgtk/data/json_config.h
        xmotion/core/camera/stereo_camera.h
        xmotion/core/camera/d_dummy_camera.h
        xmotion/fbgtk/gtk/gtk_utils.h
        xmotion/fbgtk/gtk/gtk_cam_params.h
        xmotion/fbgtk/gtk/cam_params_window.h
        xmotion/fbgtk/gtk/gtk_config_stack.h
        xmotion/core/algo/i_logic.h
        xmotion/core/algo/calibration.h
        xmotion/core/algo/pose.h
        xmotion/core/utils/cv_utils.h
        xmotion/fbgtk/data/json_ocv.h
        xmotion/core/algo/chain.h
        xmotion/core/utils/epi_util.h
        xmotion/core/algo/compose.h
        xmotion/core/filter/i_filter.h
        xmotion/core/filter/chroma_key.h

        sources/core/velocity_filter.cpp
        sources/core/low_pass_filter.cpp
        sources/core/pose_detector.cpp
        sources/core/blaze_pose.cpp
        sources/core/dnn_common.cpp
        sources/core/ssd_anchors.cpp
        sources/core/pose_roi.cpp
        sources/core/thread_pool.cpp
        sources/core/timer.cpp
        sources/core/delta_loop.cpp
        sources/core/eox_globals.cpp
        sources/fbgtk/file_boot.cpp
        sources/imgui/imgui_boot.cpp
        sources/core/texture_1.cpp
        sources/core/simple_shader.cpp
        sources/fbgtk/gl_image.cpp
        sources/fbgtk/simple_image_window.cpp
        sources/fbgtk/small_button.cpp
        sources/fbgtk/json_config.cpp
        sources/core/stereo_camera.cpp
        sources/core/a_updated_boot.cpp
        sources/fbgtk/gtk_cam_params.cpp
        sources/fbgtk/cam_params_window.cpp
        sources/fbgtk/gtk_config_stack.cpp
        sources/fbgtk/file_boot_gui.cpp
        sources/fbgtk/file_boot_cam.cpp
        sources/core/calibration.cpp
        sources/core/pose.cpp
        sources/fbgtk/file_boot_logic.cpp
        sources/core/cv_utils.cpp
        sources/fbgtk/json_ocv.cpp
        sources/core/chain.cpp
        sources/core/pose_pipeline.cpp
        sources/core/pose_pipeline_debug.cpp
        sources/core/pose_pipeline_aux.cpp
        sources/core/d_dummy_camera.cpp
        sources/core/pose_aux.cpp
        sources/core/epi_util.cpp
        sources/core/chroma_key.cpp
        sources/fbgtk/file_boot_filters.cpp
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