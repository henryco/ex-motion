//
// Created by henryco on 6/2/24.
//

#ifndef XMOTION_I_FILTER_H
#define XMOTION_I_FILTER_H

#include <opencv2/core/mat.hpp>
#include "../ocl/ocl_data.h"
#include "../ocl/ocl_interop.h"

namespace xm {

    class Filter {
    public:
        virtual cv::Mat filter(const cv::Mat &in) = 0;

        virtual cv::UMat filter(const cv::UMat &in) = 0;

        virtual xm::ocl::iop::ClImagePromise filter(const xm::ocl::Image2D &in) = 0;

        virtual ~Filter() = default;
    };

} // xm

#endif //XMOTION_I_FILTER_H
