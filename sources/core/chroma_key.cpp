//
// Created by henryco on 6/2/24.
//

#include <opencv2/imgproc.hpp>
#include "../../xmotion/core/filter/chroma_key.h"
#include "../../xmotion/core/ocl/ocl_filters.h"
#include "../../xmotion/core/utils/cv_utils.h"

namespace xm::chroma {

    void ChromaKey::init(const Conf &conf) {
        const auto key_hls = xm::ocv::bgr_to_hls(conf.key);

        const int bot_h = key_hls[0] - (int) (conf.range[0] * 255.f);
        const int bot_s = std::clamp(key_hls[2] - (int) (conf.range[1] * 255.f), 0, 255);
        const int bot_l = std::clamp(key_hls[1] - (int) (conf.range[2] * 255.f), 0, 255);

        const int top_h = key_hls[0] + (int) (conf.range[0] * 255.f);
        const int top_s = std::clamp(key_hls[2] + (int) (conf.range[1] * 255.f), 0, 255);
        const int top_l = std::clamp(key_hls[1] + (int) (conf.range[2] * 255.f), 0, 255);

        hls_key_lower = cv::Scalar(bot_h < 0 ? 255 + bot_h : bot_h, bot_l, bot_s);
        hls_key_upper = cv::Scalar(top_h > 255 ? top_h - 255 : top_h, top_l, top_s);

        mask_size = (conf.power + 1) * 256;
        blur_kernel = (conf.blur * 2) + 1;
        fine_kernel = std::max(3, (conf.fine * 2) + 1);
        mask_iterations = conf.refine;
        bgr_bg_color = conf.color;

        log->info("L: {}, {}, {}", hls_key_lower[0], hls_key_lower[1], hls_key_lower[2]);
        log->info("U: {}, {}, {}", hls_key_upper[0], hls_key_upper[1], hls_key_upper[2]);

        ready = true;
    }

    cv::Mat ChromaKey::filter(const cv::Mat &in) {
        if (!ready)
            throw std::logic_error("Filter is not initialized");
        const auto t0 = std::chrono::system_clock::now();

        cv::Mat img;

        const auto ratio = (float) in.cols / (float) in.rows;
        const auto n_w = mask_size;
        const auto n_h = (float) n_w / ratio;
        cv::resize(in, img, cv::Size((int) n_w, (int) n_h), 0, 0, cv::INTER_NEAREST);

        if (blur_kernel >= 3) {
            cv::GaussianBlur(img, img, cv::Size(blur_kernel, blur_kernel), 0, 0);
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

        cv::resize(mask, mask, in.size());
        cv::resize(mask_inv, mask_inv, in.size());

        auto bgr_front = cv::Mat();
        cv::bitwise_and(in, in, bgr_front, mask_inv);

        auto bgr_back = cv::Mat();
        auto background = cv::Mat(in.size(), in.type(), bgr_bg_color);
        cv::bitwise_and(background, background, bgr_back, mask);

        cv::Mat out;
        cv::add(bgr_front, bgr_back, out);

        const auto t1 = std::chrono::system_clock::now();
        const auto d = duration_cast<std::chrono::milliseconds>((t1 - t0)).count();

//        log->info("TC: {}", d);
        return out;
    }

    cv::UMat ChromaKey::filter(const cv::UMat &in) {
        if (!ready)
            throw std::logic_error("Filter is not initialized");
        const auto t0 = std::chrono::system_clock::now();

        cv::UMat img;
        cv::UMat out;

        const auto ratio = (float) in.cols / (float) in.rows;
        const auto n_w = mask_size;
        const auto n_h = (float) n_w / ratio;
        cv::resize(in, img, cv::Size((int) n_w, (int) n_h), 0, 0, cv::INTER_LINEAR);

        if (blur_kernel >= 3 && blur_kernel <= 31) {
            xm::ocl::blur(img, img, blur_kernel);
        }

        auto mask = cv::UMat();
        xm::ocl::bgr_in_range_hls(hls_key_lower, hls_key_upper, img, mask);

        if (mask_iterations > 0 && fine_kernel >= 3) { // morph open (reduce speckles)
            xm::ocl::erode(mask, mask, mask_iterations, fine_kernel);
            xm::ocl::dilate(mask, mask, mask_iterations, fine_kernel);
        }

        auto mask_inv = cv::UMat();
        cv::bitwise_not(mask, mask_inv);

        cv::resize(mask, mask, in.size());
        cv::resize(mask_inv, mask_inv, in.size());

        auto bgr_front = cv::UMat();
        cv::bitwise_and(in, in, bgr_front, mask_inv);

        auto bgr_back = cv::UMat();
        cv::UMat background(in.rows, in.cols, in.type(), bgr_bg_color, cv::USAGE_ALLOCATE_DEVICE_MEMORY);

        cv::bitwise_and(background, background, bgr_back, mask);

        cv::add(bgr_front, bgr_back, out);

        const auto t1 = std::chrono::system_clock::now();
        const auto d = duration_cast<std::chrono::nanoseconds>((t1 - t0)).count();
        log->info("TG: {}", d);

//        cv::cvtColor(mask, out, cv::COLOR_GRAY2BGR);
        return out;
    }

} // xm