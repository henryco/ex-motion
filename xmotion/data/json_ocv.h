//
// Created by henryco on 4/25/24.
//

#ifndef XMOTION_JSON_OCV_H
#define XMOTION_JSON_OCV_H

#include <opencv2/core/mat.hpp>

namespace xm::data::ocv {

    typedef struct {
        std::string type;

        std::string name;

        std::string timestamp;

        /**
         * Calibration matrix 3x3
         */
        cv::Mat K;

        /**
         * Distortion coefficients
         */
        cv::Mat D;

        double error;
    } Calibration;

    void write_calibration(const std::string &file, const Calibration &c);

    Calibration read_calibration(const std::string &file);

    std::string utc_iso_date_str_now();
}

#endif //XMOTION_JSON_OCV_H
