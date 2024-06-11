//
// Created by henryco on 11/06/24.
//

#ifndef XMOTION_OCL_INTEROP_H
#define XMOTION_OCL_INTEROP_H

#include "ocl_data.h"
#include <opencv2/core/mat.hpp>

namespace xm::ocl::iop {

    cv::AccessFlag access_to_cv(ACCESS access);

    inline xm::ocl::Image2D from_cv_umat(
            const cv::UMat &mat,
            xm::ocl::ACCESS access = ACCESS::RW);

    inline xm::ocl::Image2D from_cv_umat(
            const cv::UMat &mat,
            cl_context context,
            cl_device_id device,
            xm::ocl::ACCESS access = ACCESS::RW);

    /**
     * @param cv_type if -1, CV_8UC(image.channels) is used
     */
    inline cv::UMat to_cv_umat(const xm::ocl::Image2D &image, int cv_type = -1);

    /**
     * @param cv_type if -1, CV_8UC(image.channels) is used
     */
    inline void to_cv_umat(const xm::ocl::Image2D &image, cv::UMat &out, int cv_type = -1);
}

#endif //XMOTION_OCL_INTEROP_H
