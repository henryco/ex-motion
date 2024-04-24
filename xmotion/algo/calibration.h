//
// Created by henryco on 4/22/24.
//

#pragma clang diagnostic push
#pragma ide diagnostic ignored "modernize-use-nodiscard"

#ifndef XMOTION_CALIBRATION_H
#define XMOTION_CALIBRATION_H

#include "i_logic.h"
#include "../utils/timer.h"

namespace xm::calib {
    typedef struct {
        int remains_cap;
        int remains_ms;
        bool ready;

        // TODO: CALIBRATION MATRIX ETC
    } Result;

    typedef struct Initial {
        int delay = 1000;
        int total = 10;
        int columns = 9;
        int rows = 7;
        float size = 30;
        int width = 0;
        int height = 0;

        float fx = -1;
        float fy = -1;
        float cx = -1;
        float cy = -1;
        bool fix = false;
    } Initial;
}

namespace xm {

    class Calibration : public xm::Logic {

    private:
        std::vector<std::vector<cv::Point2f>> image_points{};
        std::vector<cv::Mat> images{};
        xm::calib::Result results{};
        xm::calib::Initial config;
        eox::utils::Timer timer{};

        bool active = false;

    public:
        void init(const xm::calib::Initial &params);

        Calibration &proceed(float delta, const std::vector<cv::Mat> &frames) override;

        bool is_active() const override;

        void start() override;

        void stop() override;

        const std::vector<cv::Mat> &frames() const override;

        const xm::calib::Result &result() const;

    private:
        bool capture_squares(const cv::Mat &frame);
    };

} // xm

#endif //XMOTION_CALIBRATION_H

#pragma clang diagnostic pop