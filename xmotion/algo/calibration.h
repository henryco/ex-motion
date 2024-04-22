//
// Created by henryco on 4/22/24.
//

#pragma clang diagnostic push
#pragma ide diagnostic ignored "modernize-use-nodiscard"

#ifndef XMOTION_CALIBRATION_H
#define XMOTION_CALIBRATION_H

#include "i_logic.h"

namespace xm::calib {
    typedef struct {
        // TODO
    } Result;
}

namespace xm {

    class Calibration : public xm::Logic {

    private:
        std::vector<cv::Mat> images{};
        xm::calib::Result results{};
        bool active = false;

    public:
        Calibration &proceed(float delta, const std::vector<cv::Mat> &frames) override;

        bool is_active() const override;

        void start() override;

        void stop() override;

        const std::vector<cv::Mat> &frames() const override;

        const xm::calib::Result &result() const;
    };

} // xm

#endif //XMOTION_CALIBRATION_H

#pragma clang diagnostic pop