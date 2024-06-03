//
// Created by henryco on 6/2/24.
//

#include <opencv2/imgproc.hpp>
#include "../../xmotion/core/filter/chroma_key.h"
#include "../../xmotion/core/utils/cv_utils.h"

namespace xm::chroma {

    void ChromaKey::init(const Conf &conf) {
        const auto key_hls = xm::ocv::bgr_to_hls(conf.key);

        const int bot_h = std::clamp(key_hls[0] - (int) (conf.range[0] * 255.), 0, 255);
        const int bot_s = std::clamp(key_hls[2] - (int) (conf.range[1] * 255.), 0, 255);
        const int bot_l = std::clamp(key_hls[1] - (int) (conf.range[2] * 255.), 0, 255);

        const int top_h = std::clamp(key_hls[0] + (int) (conf.range[0] * 255.), 0, 255);
        const int top_s = std::clamp(key_hls[2] + (int) (conf.range[1] * 255.), 0, 255);
        const int top_l = std::clamp(key_hls[1] + (int) (conf.range[2] * 255.), 0, 255);

        hls_key_lower = cv::Scalar(bot_h, bot_l, bot_s);
        hls_key_upper = cv::Scalar(top_h, top_l, top_s);

        blur_kernel = (conf.blur * 2) + 1;
        mask_iterations = conf.refine;
        bgr_bg_color = conf.color;

        log->info("L: {}, {}, {}", hls_key_lower[0], hls_key_lower[1], hls_key_lower[2]);
        log->info("U: {}, {}, {}", hls_key_upper[0], hls_key_upper[1], hls_key_upper[2]);

        ready = true;
    }

    cv::Mat ChromaKey::filter(const cv::Mat &in) {
        if (!ready)
            throw std::logic_error("Filter is not initialized");

        cv::Mat img;
        if (blur_kernel >= 3) {
            cv::GaussianBlur(in, img, cv::Size(blur_kernel, blur_kernel), 0, 0);
        } else {
            img = in;
        }

        auto hls_image = cv::Mat();
        cv::cvtColor(img, hls_image, cv::COLOR_BGR2HLS_FULL);

        auto mask = cv::Mat();
        cv::inRange(hls_image, hls_key_lower, hls_key_upper, mask);

        if (mask_iterations > 0) {
            cv::erode(mask, mask, cv::Mat(), cv::Point(-1, -1), mask_iterations);
            cv::dilate(mask, mask, cv::Mat(), cv::Point(-1, -1), mask_iterations);
        }

        auto mask_inv = cv::Mat();
        cv::bitwise_not(mask, mask_inv);

        auto bgr_front = cv::Mat();
        cv::bitwise_and(in, in, bgr_front, mask_inv);

        auto bgr_back = cv::Mat();
        auto background = cv::Mat(in.size(), in.type(), bgr_bg_color);
        cv::bitwise_and(background, background, bgr_back, mask);

        cv::Mat out;
        cv::add(bgr_front, bgr_back, out);
        return out;
    }

    cv::UMat ChromaKey::filter(const cv::UMat &in) {
        return cv::UMat();
    }

} // xm