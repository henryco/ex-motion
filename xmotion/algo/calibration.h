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

    class Calibration : public xm::Logic<xm::calib::Result> {

    public:
        xm::calib::Result proceed(const std::vector<cv::Mat> &frames) override;

        std::vector<cv::Mat> frames() const override;

        void start() override;

        void stop() override;
    };

} // xm

#endif //XMOTION_CALIBRATION_H

#pragma clang diagnostic pop