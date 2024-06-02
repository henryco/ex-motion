//
// Created by henryco on 6/2/24.
//

#ifndef XMOTION_A_FILTER_H
#define XMOTION_A_FILTER_H

#include <opencv2/core/mat.hpp>
namespace xm {

    class Filter {
    protected:
        bool u_mat = false;

        int m_type;
        cv::Size_<int> m_size;

        [[nodiscard]] cv::_InputOutputArray new_mat() const;

        [[nodiscard]] cv::_InputOutputArray from_mat(const cv::Mat &in) const;

        [[nodiscard]] cv::_InputOutputArray from_any(cv::InputArray &in) const;

        [[nodiscard]] cv::Size_<int> curr_size() const;

        [[nodiscard]] int curr_type() const;

    public:
        virtual void filter(cv::InputArray in, cv::OutputArray out) = 0;

        cv::Mat filter(const cv::Mat &in);

        cv::UMat filter(const cv::UMat &in);

        virtual ~Filter() = default;
    };

} // xm

#endif //XMOTION_A_FILTER_H
