//
// Created by henryco on 6/5/24.
//

#ifndef XMOTION_DNN_CL_UTILS_H
#define XMOTION_DNN_CL_UTILS_H

#include <opencv2/core/mat.hpp>
#include <CL/cl.h>

namespace xm::dnn::ocl {

    cl_mem getOpenCLBufferFromUMat(cv::UMat& umat);

    cl_mem getOpenCLBufferFromUMat(const cv::UMat& umat);

    void checkGLSharingSupport();

} // xm

#endif //XMOTION_DNN_CL_UTILS_H
