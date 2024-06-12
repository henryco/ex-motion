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

    QueuePromise::QueuePromise(const Image2D &out,
                               cl_command_queue _queue,
                               cl_event _event):
            image(out),
            ocl_queue(_queue),
            ocl_event(_event),
            completed(false){}

    void QueuePromise::toUMat(cv::UMat &mat) {
        xm::ocl::iop::to_cv_umat(image, mat);
    }

    cv::UMat QueuePromise::getUMat() {
        return xm::ocl::iop::to_cv_umat(image);
    }

    xm::ocl::Image2D QueuePromise::getImage2D() {
        return image;
    }

    QueuePromise &QueuePromise::waitFor() {
        if (completed)
            return *this;
        {
            cl_int err;
            err = clFinish(ocl_queue);
            if (err != CL_SUCCESS)
                throw std::runtime_error("Cannot finish command queue: " + std::to_string(err));
            completed = true;
        }
        return *this;
    }

    QueuePromise::~QueuePromise() {
        if (ocl_event != nullptr)
            clReleaseEvent(ocl_event);
    }

    bool QueuePromise::resolved() const {
        return completed;
    }

    const cl_event &QueuePromise::event() const {
        return ocl_event;
    }

    QueuePromise::QueuePromise(QueuePromise &&other) noexcept {
        completed = other.completed;
        ocl_event = other.ocl_event;
        ocl_queue = other.ocl_queue;
        image = std::move(other.image);
        other.ocl_event = nullptr;
        other.ocl_queue = nullptr;
        other.completed = true;
    }

    QueuePromise::QueuePromise(const QueuePromise &other) {
        completed = other.completed;
        ocl_queue = other.ocl_queue;
        ocl_event = other.ocl_event;
        image = other.image;
        clRetainEvent(ocl_event);
    }

    QueuePromise &QueuePromise::operator=(QueuePromise &&other) noexcept {
        if (this == &other)
            return *this;
        if (ocl_event != nullptr)
            clReleaseEvent(ocl_event);
        completed = other.completed;
        ocl_queue = other.ocl_queue;
        ocl_event = other.ocl_event;
        image = std::move(other.image);
        other.ocl_event = nullptr;
        other.ocl_queue = nullptr;
        other.completed = true;
        return *this;
    }

    QueuePromise &QueuePromise::operator=(const QueuePromise &other) {
        if (this == &other)
            return *this;
        if (ocl_event != nullptr)
            clReleaseEvent(ocl_event);
        completed = other.completed;
        ocl_event = other.ocl_event;
        ocl_queue = other.ocl_queue;
        image = other.image;
        clRetainEvent(ocl_event);
        return *this;
    }

    cl_command_queue QueuePromise::queue() const {
        return ocl_queue;
    }
}