//
// Created by henryco on 6/2/24.
//

#include "../../xmotion/core/filter/a_filter.h"

namespace xm {

    cv::Mat Filter::filter(const cv::Mat &in) {
        cv::Mat out;
        u_mat = false;
        m_size = in.size();
        m_type = in.type();
        filter(in, out);
        u_mat = false;
        return out;
    }

    cv::UMat Filter::filter(const cv::UMat &in) {
        cv::UMat out;
        u_mat = true;
        m_size = in.size();
        m_type = in.type();
        filter(in, out);
        u_mat = false;
        return out;
    }

    cv::_InputOutputArray Filter::new_mat() const {
        if (u_mat)
            return cv::UMat();
        return cv::Mat();
    }

    cv::_InputOutputArray Filter::from_mat(const cv::Mat &in) const {
        if (u_mat) {
            cv::UMat mat;
            in.copyTo(mat);
            return mat;
        }
        return in;
    }

    cv::_InputOutputArray Filter::from_any(cv::InputArray &in) const {
        if (u_mat) {
            cv::UMat out;
            out.setTo(in);
            return out;
        }
        cv::Mat out;
        out.setTo(in);
        return out;
    }

    cv::Size_<int> Filter::curr_size() const {
        return m_size;
    }

    int Filter::curr_type() const {
        return m_type;
    }
}