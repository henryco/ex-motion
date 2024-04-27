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

        /**
        * aperture width of the sensor
        */
        float width;

        /**
         * aperture height of the sensor
         */
        float height;

        /**
         * Field of view along X axis
         */
        float fov_x;

        /**
         * Field of view along Y axis
         */
        float fov_y;

        /**
         * Focal length of the lense
         */
        float f;

        /**
         * Principal point X
         */
        float c_x;

        /**
         * Principal point Y
         */
        float c_y;

        /**
         * Aspect ratio fy/fx
         */
        float r;

        /**
         * Measurement error
         */
        double error;
    } Calibration;

    void write_calibration(const std::string &file, const Calibration &c);

    Calibration read_calibration(const std::string &file);

    std::string utc_iso_date_str_now();
}

#endif //XMOTION_JSON_OCV_H
