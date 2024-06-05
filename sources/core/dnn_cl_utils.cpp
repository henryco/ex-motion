//
// Created by henryco on 6/5/24.
//

#include "../../xmotion/core/dnn/net/dnn_cl_utils.h"
#include <iostream>

namespace xm::dnn::ocl {

    cl_mem getOpenCLBufferFromUMat(cv::UMat& umat) {
        return (cl_mem)umat.handle(cv::ACCESS_RW);
    }

    cl_mem getOpenCLBufferFromUMat(const cv::UMat &umat) {
        return (cl_mem)umat.handle(cv::ACCESS_READ);
    }

    void checkGLSharingSupport() {
        cl_int err;

        // Get the number of platforms
        cl_uint numPlatforms;
        err = clGetPlatformIDs(0, NULL, &numPlatforms);
        if (err != CL_SUCCESS) {
            std::cerr << "Failed to get number of OpenCL platforms: " << err << std::endl;
            return;
        }

        // Get the platform IDs
        std::vector<cl_platform_id> platforms(numPlatforms);
        err = clGetPlatformIDs(numPlatforms, platforms.data(), NULL);
        if (err != CL_SUCCESS) {
            std::cerr << "Failed to get OpenCL platforms: " << err << std::endl;
            return;
        }

        // Iterate over each platform
        for (cl_uint i = 0; i < numPlatforms; ++i) {
            // Get the number of devices for this platform
            cl_uint numDevices;
            err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
            if (err != CL_SUCCESS) {
                std::cerr << "Failed to get number of OpenCL devices: " << err << std::endl;
                continue;
            }

            // Get the device IDs
            std::vector<cl_device_id> devices(numDevices);
            err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, numDevices, devices.data(), NULL);
            if (err != CL_SUCCESS) {
                std::cerr << "Failed to get OpenCL devices: " << err << std::endl;
                continue;
            }

            // Iterate over each device
            for (cl_uint j = 0; j < numDevices; ++j) {
                // Get the extensions string size
                size_t extensionSize;
                err = clGetDeviceInfo(devices[j], CL_DEVICE_EXTENSIONS, 0, NULL, &extensionSize);
                if (err != CL_SUCCESS) {
                    std::cerr << "Failed to get OpenCL device extension size: " << err << std::endl;
                    continue;
                }

                // Get the extensions string
                std::vector<char> extensions(extensionSize);
                err = clGetDeviceInfo(devices[j], CL_DEVICE_EXTENSIONS, extensionSize, extensions.data(), NULL);
                if (err != CL_SUCCESS) {
                    std::cerr << "Failed to get OpenCL device extensions: " << err << std::endl;
                    continue;
                }

                // Check if cl_khr_gl_sharing is supported
                std::string extensionsStr(extensions.begin(), extensions.end());
                if (extensionsStr.find("cl_khr_gl_sharing") != std::string::npos) {
                    std::cout << "Device " << j << " on Platform " << i << " supports cl_khr_gl_sharing" << std::endl;
                } else {
                    std::cout << "Device " << j << " on Platform " << i << " does not support cl_khr_gl_sharing" << std::endl;
                }
            }
        }
    }

} // xm