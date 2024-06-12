//
// Created by henryco on 11/06/24.
//

#ifndef XMOTION_OCL_INTEROP_H
#define XMOTION_OCL_INTEROP_H

#include "ocl_data.h"
#include <opencv2/core/mat.hpp>

namespace xm::ocl::iop {

    cv::AccessFlag access_to_cv(ACCESS access);

    xm::ocl::Image2D from_cv_umat(
            const cv::UMat &mat,
            xm::ocl::ACCESS access = ACCESS::RW);

    xm::ocl::Image2D from_cv_umat(
            const cv::UMat &mat,
            cl_context context,
            cl_device_id device,
            xm::ocl::ACCESS access = ACCESS::RW);

    /**
     * @param cv_type if -1, CV_8UC(image.channels) is used
     */
    cv::UMat to_cv_umat(const xm::ocl::Image2D &image, int cv_type = -1);

    /**
     * @param cv_type if -1, CV_8UC(image.channels) is used
     */
    void to_cv_umat(const xm::ocl::Image2D &image, cv::UMat &out, int cv_type = -1);


    /**
     * Not really thread safe!!!
     */
    class QueuePromise {
    private:
        cl_command_queue ocl_queue = nullptr;
        cl_event ocl_event = nullptr;
        xm::ocl::Image2D image;
        bool completed = false;

    public:
        QueuePromise(const xm::ocl::Image2D &out,
                     cl_command_queue ocl_queue,
                     cl_event ocl_event = nullptr);

        ~QueuePromise();

        QueuePromise(QueuePromise &&other) noexcept;

        QueuePromise(const QueuePromise &other);

        QueuePromise &operator=(QueuePromise &&other) noexcept;

        QueuePromise &operator=(const QueuePromise &other);

        /**
         * Often you should call waitFor() first
         */
        xm::ocl::Image2D getImage2D();

        /**
         * Often you should call waitFor() first
         */
        void toUMat(cv::UMat &mat);

        /**
         * Often you should call waitFor() first
         */
        cv::UMat getUMat();

        /**
         * Waits for data to be ready
         */
        QueuePromise &waitFor();

        bool resolved() const;

        const cl_event &event() const;

        cl_command_queue queue() const;
    };
}

#endif //XMOTION_OCL_INTEROP_H
