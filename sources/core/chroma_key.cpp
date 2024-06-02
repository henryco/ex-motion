//
// Created by henryco on 6/2/24.
//

#include <opencv2/imgproc.hpp>
#include "../../xmotion/core/filter/chroma_key.h"
#include "../../xmotion/core/utils/cv_utils.h"

namespace xm::chroma {

    void ChromaKey::init(const Conf &conf) {
        const auto key_hsl = xm::ocv::bgr_to_hls(conf.key);

        const auto bot_h = std::clamp(key_hsl[0] - (conf.range[0] * 255.), 0., 255.);
        const auto bot_s = std::clamp(key_hsl[1] - (conf.range[1] * 255.), 0., 255.);
        const auto bot_l = std::clamp(key_hsl[2] - (conf.range[2] * 255.), 0., 255.);

        const auto top_h = std::clamp(key_hsl[0] + (conf.range[0] * 255.), 0., 255.);
        const auto top_s = std::clamp(key_hsl[1] + (conf.range[1] * 255.), 0., 255.);
        const auto top_l = std::clamp(key_hsl[2] + (conf.range[2] * 255.), 0., 255.);

        hls_key_lower = cv::Scalar(bot_h, bot_l, bot_s);
        hls_key_upper = cv::Scalar(top_h, top_l, top_s);
        mask_iterations = conf.refine;
        bgr_bg_color = conf.color;

        up_to_date = false;
        ready = true;
    }

    void ChromaKey::filter(cv::InputArray in, cv::OutputArray out) {
        if (!ready)
            throw std::logic_error("Filter is not initialized");

        if (!up_to_date) {
            background = from_mat(cv::Mat(curr_size(), curr_type(), bgr_bg_color));
            up_to_date = true;
        }

        log->info("W: {}, H: {}", curr_size().width, curr_size().height);
        log->info("w: {}, h: {}", in.getSz().width, in.getSz().height);

        auto hls_image = new_mat();
        cv::cvtColor(in, hls_image, cv::COLOR_BGR2HLS_FULL);

        auto mask = new_mat();
        cv::inRange(hls_image, from_any(hls_key_lower), from_any(hls_key_upper), mask);

        if (mask_iterations > 0) {
            cv::erode(mask, mask, new_mat(), cv::Point(-1, -1), mask_iterations);
            cv::dilate(mask, mask, new_mat(), cv::Point(-1, -1), mask_iterations);
        }

        auto mask_inv = new_mat();
        cv::bitwise_not(mask, mask_inv);

        auto bgr_front = new_mat();
        cv::bitwise_and(in, in, bgr_front, mask_inv);

        auto bgr_back = new_mat();
        cv::bitwise_and(background, background, bgr_back, mask);

        cv::add(bgr_front, bgr_back, out);
    }

} // xm