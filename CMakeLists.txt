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


add_executable(xmotion src/main.cpp)


if (WIN32)
    # Include directories for the specific target
    target_include_directories(${PROJECT_NAME}
            PRIVATE ${OpenCV_INCLUDE_DIRS}
            PRIVATE ${GTKMM_INCLUDE_DIRS})

    # Windows media foundations framework
    target_link_libraries(${PROJECT_NAME} mfplat mf mfreadwrite)

elseif (UNIX AND NOT APPLE)
    # Include directories for the specific target
    target_include_directories(${PROJECT_NAME}
            PRIVATE ${OpenCV_INCLUDE_DIRS}
            PRIVATE ${GTKMM_INCLUDE_DIRS}
            PRIVATE ${V4L2_INCLUDE_DIRS})
endif ()

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