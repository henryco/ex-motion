cmake_minimum_required(VERSION 3.27)
project(xmotion VERSION 0.2 LANGUAGES CXX)

# By default support for nvidia is disabled
option(WITH_NVIDIA "Build for NVidia gpu" OFF)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(TFLITE_ENABLE_GPU ON CACHE BOOL "Enable TFLite GPU support")
set(ABSL_PROPAGATE_CXX_STD ON)
set(JSON_BuildTests OFF CACHE INTERNAL "")

# This may cause compatibility issues ====
set(CL_HPP_TARGET_OPENCL_VERSION 300)
set(CL_TARGET_OPENCL_VERSION 300)
# ========================================

if (WITH_NVIDIA)
    add_subdirectory(external/onnxruntime/cmake)

    set(onnxruntime_USE_CUDA ON)
    set(onnxruntime_BUILD_SHARED_LIB ON)
    set(onnxruntime_CUDA_HOME ${CUDA_TOOLKIT_ROOT_DIR})

    include_directories(${CUDA_INCLUDE_DIRS})
else ()
    add_subdirectory(external/json)
endif ()

add_subdirectory(external/tensorflow/tensorflow/lite)
add_subdirectory(external/argparse)
add_subdirectory(external/glm)

set(CMAKE_C_COMPILER "clang-18")
set(CMAKE_CXX_COMPILER "clang++-18")

# Creates the compile_commands.json used by editors for code completion
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Must use GNUInstallDirs to install libraries into correct locations on all platforms:
include(GNUInstallDirs)

# Find OpenCL 3.0
find_package(OpenCL 3.0 REQUIRED)
if (OpenCL_FOUND)
    add_definitions("-DCL_HPP_TARGET_OPENCL_VERSION=300")
    add_definitions("-DCL_TARGET_OPENCL_VERSION=300")
endif ()

# Find OpenCL headers
find_package(OpenCLHeaders REQUIRED)

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

set(KERNEL_FILES
        kernels/subsense.h
        kernels/chroma_key.h
        kernels/flip_rotate.h
        kernels/color_space.h
        kernels/filter_conv.h
        kernels/letter_box.h
)

set(HEADER_FILES
        platforms/agnostic_cap.h
        xmotion/core/utils/velocity_filter.h
        xmotion/core/utils/low_pass_filter.h
        xmotion/core/dnn/net/pose_detector.h
        xmotion/core/dnn/net/dnn_runner.h
        xmotion/core/dnn/net/blaze_pose.h
        xmotion/core/dnn/net/dnn_common.h
        xmotion/core/dnn/net/ssd_anchors.h
        xmotion/core/dnn/net/pose_roi.h
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
        xmotion/core/filter/i_filter.h
        xmotion/core/filter/chroma_key.h
        xmotion/core/ocl/kernel.h
        xmotion/core/ocl/ocl_filters.h
        xmotion/core/dnn/net/dnn_cl_utils.h
        xmotion/core/ocl/cl_kernel.h
        xmotion/core/ocl/ocl_interop.h
        xmotion/core/ocl/ocl_interop_ext.h
        xmotion/fbgtk/file_worker.h
        xmotion/core/ocl/ocl_data.h
        xmotion/core/ocl/ocl_container.h
        xmotion/core/utils/xm_data.h
        xmotion/core/filter/bg_subtract.h
        xmotion/fbgtk/data/json_config_filters.h
        xmotion/fbgtk/data/json_config_pose.h
        xmotion/fbgtk/data/json_config_common.h
        xmotion/fbgtk/data/json_config_calibration.h
        xmotion/fbgtk/data/json_config_gui.h
        xmotion/core/filter/blur.h
        xmotion/core/pose/pose_agnostic.h
        xmotion/core/pose/roi/roi_body_heuristics.h
        xmotion/core/pose/detector/body_detector.h
        platforms/agnostic_dnn.h
        platforms/agnostic_detector.h
        platforms/agnostic_body.h
        xmotion/core/pose/body/body_marker.h
)

set(SOURCE_FILES
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
        sources/fbgtk/file_worker_gui.cpp
        sources/fbgtk/file_worker_cam.cpp
        sources/core/calibration.cpp
        sources/core/pose.cpp
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
        sources/fbgtk/file_worker_filters.cpp
        sources/core/kernel.cpp
        sources/core/ocl_filters.cpp
        sources/core/dnn_cl_utils.cpp
        sources/core/cl_kernel.cpp
        sources/core/ocl_data.cpp
        sources/core/ocl_interop.cpp
        sources/fbgtk/file_worker.cpp
        sources/core/ocl_container.cpp
        sources/core/xm_data.cpp
        sources/core/bg_subtract.cpp
        sources/core/blur.cpp
        sources/fbgtk/file_worker_pose.cpp
        sources/fbgtk/file_worker_calib.cpp
        sources/fbgtk/file_worker_compose.cpp
        sources/core/pose_agnostic.cpp
        sources/core/roi_body_heuristics.cpp
        sources/core/body_detector.cpp
        sources/core/body_marker.cpp
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

if (WITH_NVIDIA)

    list(APPEND SOURCE_FILES platforms/nvidia/agnostic_dnn_nvidia.h)
    list(APPEND SOURCE_FILES platforms/nvidia/agnostic_dnn_nvidia.cpp)
    list(APPEND SOURCE_FILES platforms/nvidia/agnostic_detector_nvidia.cpp)
    list(APPEND SOURCE_FILES platforms/nvidia/agnostic_body_nvidia.cpp)

else ()

    list(APPEND SOURCE_FILES platforms/tflite/agnostic_dnn_tflite.h)
    list(APPEND SOURCE_FILES platforms/tflite/agnostic_dnn_tflite.cpp)
    list(APPEND SOURCE_FILES platforms/tflite/agnostic_detector_tflite.cpp)
    list(APPEND SOURCE_FILES platforms/tflite/agnostic_body_tflite.cpp)

endif ()

# =================== OpenCL kernels generation =====================
add_executable(convert_to_c_array scripts/embed_file.cxx)
set(GENERATED_CL_SOURCES)

foreach (input_file IN LISTS KERNEL_FILES)
    get_filename_component(output_path ${input_file} DIRECTORY)
    get_filename_component(output_name ${input_file} NAME_WE)
    set(kernel_name "${PROJECT_SOURCE_DIR}/${output_path}/cl/${output_name}.cl")
    set(output_name "${PROJECT_BINARY_DIR}/xmotion_files/generated/${output_path}/${output_name}.cpp")
    add_custom_command(
            OUTPUT ${output_name}
            COMMAND convert_to_c_array ${PROJECT_SOURCE_DIR}/${input_file} ${kernel_name} ${output_name}
            DEPENDS convert_to_c_array ${PROJECT_SOURCE_DIR}/${input_file} ${kernel_name}
            COMMENT "Converting: ${input_file} -> ${output_name}"
            VERBATIM)
    list(APPEND GENERATED_CL_SOURCES ${output_name})
endforeach ()

add_custom_target(generate_cl_sources
        ALL DEPENDS ${GENERATED_CL_SOURCES})
# =================== OpenCL kernels generation =====================


set(RESULT_SOURCES
        main.cpp
        ${HEADER_FILES}
        ${SOURCE_FILES}
        ${KERNEL_FILES}
        ${GENERATED_CL_SOURCES})

add_executable(xmotion
        ${RESULT_SOURCES})

#message(STATUS
#        "FILES: ${RESULT_SOURCES}")

add_dependencies(xmotion
        generate_cl_sources)

if (WIN32)

    # Include directories for the specific target
    target_include_directories(${PROJECT_NAME}
            PRIVATE ${OpenCV_INCLUDE_DIRS}
            PRIVATE ${GTKMM_INCLUDE_DIRS})

    # Link libraries
    if (WITH_NVIDIA)
        target_link_libraries(${PROJECT_NAME}
                mfplat mf mfreadwrite
                PRIVATE onnxruntime
                PRIVATE ${CUDA_LIBRARIES}
                PRIVATE ${OpenCV_LIBS}
                PRIVATE OpenCL::OpenCL
                PRIVATE OpenCL::Headers
                PRIVATE ${GTKMM_LIBRARIES}
                PRIVATE OpenGL::GL
                PRIVATE spdlog::spdlog
                PRIVATE nlohmann_json::nlohmann_json
                PRIVATE tensorflow-lite
                PRIVATE argparse
                PRIVATE glm)
    else ()
        target_link_libraries(${PROJECT_NAME}
                mfplat mf mfreadwrite
                PRIVATE ${OpenCV_LIBS}
                PRIVATE OpenCL::OpenCL
                PRIVATE OpenCL::Headers
                PRIVATE ${GTKMM_LIBRARIES}
                PRIVATE OpenGL::GL
                PRIVATE spdlog::spdlog
                PRIVATE nlohmann_json::nlohmann_json
                PRIVATE tensorflow-lite
                PRIVATE argparse
                PRIVATE glm)
    endif ()

    target_compile_options(${PROJECT_NAME}
            PRIVATE ${GTKMM_CFLAGS_OTHER})

elseif (UNIX AND NOT APPLE)

    # Include directories for the specific target
    target_include_directories(${PROJECT_NAME}
            PRIVATE ${OpenCV_INCLUDE_DIRS}
            PRIVATE ${GTKMM_INCLUDE_DIRS}
            PRIVATE ${V4L2_INCLUDE_DIRS})

    # Link libraries
    if (WITH_NVIDIA)
        target_link_libraries(${PROJECT_NAME}
                PRIVATE onnxruntime
                PRIVATE ${CUDA_LIBRARIES}
                PRIVATE ${OpenCV_LIBS}
                PRIVATE ${GTKMM_LIBRARIES}
                PRIVATE OpenGL::GL
                PRIVATE OpenCL::OpenCL
                PRIVATE OpenCL::Headers
                PRIVATE spdlog::spdlog
                PRIVATE ${V4L2_LIBRARIES}
                PRIVATE nlohmann_json::nlohmann_json
                PRIVATE tensorflow-lite
                PRIVATE argparse
                PRIVATE glm)
    else ()
        target_link_libraries(${PROJECT_NAME}
                PRIVATE ${OpenCV_LIBS}
                PRIVATE ${GTKMM_LIBRARIES}
                PRIVATE OpenGL::GL
                PRIVATE OpenCL::OpenCL
                PRIVATE OpenCL::Headers
                PRIVATE spdlog::spdlog
                PRIVATE ${V4L2_LIBRARIES}
                PRIVATE nlohmann_json::nlohmann_json
                PRIVATE tensorflow-lite
                PRIVATE argparse
                PRIVATE glm)
    endif ()

    target_compile_options(${PROJECT_NAME}
            PRIVATE ${GTKMM_CFLAGS_OTHER}
            PRIVATE ${V4L2_CFLAGS_OTHER})

endif ()

# This may cause compatibility issues ============
target_compile_definitions(${PROJECT_NAME}
        PRIVATE CL_TARGET_OPENCL_VERSION=300
        PRIVATE CL_HPP_TARGET_OPENCL_VERSION=300)
# ================================================