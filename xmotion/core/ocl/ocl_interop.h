//
// Created by henryco on 11/06/24.
//

#ifndef XMOTION_OCL_INTEROP_H
#define XMOTION_OCL_INTEROP_H

#include <opencv2/core/mat.hpp>
#include <functional>

#include "ocl_data.h"
#include "ocl_interop_ext.h"
#include "ocl_container.h"

namespace xm::ocl::iop {

    /**
     * Not really thread safe!!!
     * (but who cares, it don't really have to)
     */
    class ClImagePromise {
    private:
        std::shared_ptr<ResourceContainer> cleanup_container = nullptr;
        cl_command_queue ocl_queue = nullptr;
        cl_event ocl_event = nullptr;
        xm::ocl::Image2D image;
        bool completed = false;

    public:
        ClImagePromise(const xm::ocl::Image2D &out,
                       cl_command_queue ocl_queue,
                       cl_event ocl_event = nullptr);

        ClImagePromise(const xm::ocl::Image2D &out, // NOLINT(*-explicit-constructor)
                       cl_event ocl_event = nullptr);

        ClImagePromise() = default;

        ~ClImagePromise();

        ClImagePromise(ClImagePromise &&other) noexcept;

        ClImagePromise(const ClImagePromise &other);

        ClImagePromise &operator=(ClImagePromise &&other) noexcept;

        ClImagePromise &operator=(const ClImagePromise &other);

        ClImagePromise &withCleanup(std::function<void()> *cb_ptr);

        ClImagePromise &withCleanup(const ClImagePromise &other);

        /**
          * Waits for data to be ready
          */
        ClImagePromise &waitFor();

        /**
         * Often you should call waitFor() first
         */
        void toImage2D(xm::ocl::Image2D &img) const;

        /**
         * Often you should call waitFor() first
         */
        xm::ocl::Image2D getImage2D() const;

        /**
         * Often you should call waitFor() first
         */
        void toUMat(cv::UMat &mat) const;

        /**
         * Often you should call waitFor() first
         */
        cv::UMat getUMat() const;

        /**
         * Often you should call waitFor() first.
         * Also Painfully slow and inefficient!
         */
        void toMat(cv::Mat &mat) const;

        /**
         * Often you should call waitFor() first.
         * Also Painfully slow and inefficient!
         */
        cv::Mat getMat() const;

        bool resolved() const;

        const cl_event &event() const;

        cl_command_queue queue() const;

        /**
         * BLOCKING OPERATION, waits for result and cleanup resources
         * @param promises list of unique promises
         * @param force wait for completed promises too
         */
        static void finalizeAll(std::vector<ClImagePromise> &promises, bool force = false);
    };

    cv::AccessFlag access_to_cv(ACCESS access);

    cl_mem_flags access_to_cl(ACCESS access);

    xm::ocl::Image2D from_cv_mat(
            const cv::Mat &mat,
            xm::ocl::ACCESS access = ACCESS::RW);

    xm::ocl::Image2D from_cv_mat(
            const cv::Mat &mat,
            cl_context context,
            cl_device_id device,
            xm::ocl::ACCESS access = ACCESS::RW);

    ClImagePromise from_cv_mat(
            const cv::Mat &mat,
            cl_command_queue queue,
            xm::ocl::ACCESS access = ACCESS::RW);

    ClImagePromise from_cv_mat(
            const cv::Mat &mat,
            cl_context context,
            cl_device_id device,
            cl_command_queue queue,
            xm::ocl::ACCESS access = ACCESS::RW);

    xm::ocl::Image2D from_cv_umat(
            const cv::UMat &mat,
            xm::ocl::ACCESS access = ACCESS::RW);

    xm::ocl::Image2D from_cv_umat(
            const cv::UMat &mat,
            cl_context context,
            cl_device_id device,
            xm::ocl::ACCESS access = ACCESS::RW);

    ClImagePromise copy_ocl(
            const xm::ocl::Image2D &image,
            cl_command_queue queue,
            xm::ocl::ACCESS access = ACCESS::RW);

    ClImagePromise copy_ocl(
            const xm::ocl::Image2D &image,
            cl_command_queue queue,
            int xo, int yo, int width, int height,
            xm::ocl::ACCESS access = ACCESS::RW);

    /**
     * @param cv_type if -1, CV_8UC(image.channels) is used
     */
    cv::UMat to_cv_umat(const xm::ocl::Image2D &image, int cv_type = -1);

    /**
     * @param cv_type if -1, CV_8UC(image.channels) is used
     */
    void to_cv_umat(const xm::ocl::Image2D &image, cv::UMat &out, int cv_type = -1);

    void to_cv_mat(const xm::ocl::Image2D &image, cv::Mat &out, cl_command_queue queue, int cv_type = -1);

    CLPromise<cv::Mat> to_cv_mat(const xm::ocl::Image2D &image, cl_command_queue queue, int cv_type = -1);
}

#endif //XMOTION_OCL_INTEROP_H
