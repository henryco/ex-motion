//
// Created by henryco on 11/06/24.
//

#include <opencv2/core/ocl.hpp>
#include "../../xmotion/core/ocl/ocl_interop.h"

namespace xm::ocl::iop {

    cv::AccessFlag access_to_cv(ACCESS access) {
        if (access == ACCESS::RW)
            return cv::ACCESS_RW;
        if (access == ACCESS::RO)
            return cv::ACCESS_READ;
        if (access == ACCESS::WO)
            return cv::ACCESS_WRITE;
        throw std::invalid_argument("Unknown access modifier");
    }

    xm::ocl::Image2D from_cv_umat(const cv::UMat &source, ACCESS modifier) {
        return from_cv_umat(
                source,
                (cl_context) cv::ocl::Context::getDefault().ptr(),
                (cl_device_id) cv::ocl::Device::getDefault().ptr(),
                modifier);
    }

    xm::ocl::Image2D from_cv_umat(const cv::UMat &mat, cl_context context, cl_device_id device, ACCESS access) {
        return xm::ocl::Image2D(
                mat.cols,
                mat.rows,
                (size_t) mat.channels(),
                mat.elemSize1(),
                (cl_mem) mat.handle(access_to_cv(access)),
                context,
                device,
                access).retain();
    }

    cv::UMat to_cv_umat(const Image2D &image, int cv_type) {
        cv::UMat mat;
        to_cv_umat(image, mat, cv_type);
        return mat;
    }

    void to_cv_umat(const Image2D &image, cv::UMat &out, int cv_type) {
        cv::ocl::convertFromBuffer(image.handle,
                                   image.channels * image.cols,
                                   (int) image.rows,
                                   (int) image.cols,
                                   (int) (cv_type < 0 ? (CV_8UC(image.channels)) : cv_type),
                                   out);
    }
}