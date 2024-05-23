//
// Created by henryco on 4/25/24.
//

#ifndef XMOTION_JSON_OCV_H
#define XMOTION_JSON_OCV_H

#include <opencv2/core/mat.hpp>

namespace xm::data::ocv {

    typedef struct Calibration {
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

    typedef struct ChainCalibration {
        std::string type;

        std::string name;

        std::string timestamp;

        /**
         * Rotation matrix 3x3
         */
        cv::Mat R;

        /**
         * Translation vector
         */
        cv::Mat T;

        /**
         * Essential matrix
         */
        cv::Mat E;

        /**
         * Fundamental matrix
         */
        cv::Mat F;

        /**
         * [R|t] basis change 4x4 homogeneous matrix
         * according to previous camera within the chain.
         * \code
         * ┌ R R R tx ┐
         * │ R R R ty │
         * │ R R R tz │
         * └ 0 0 0 1  ┘
         * \endcode
         */
        cv::Mat RT;

        /**
         * Same as [R|t] matrix, but
         * according to first camera within the chain.
         * \ref CrossCalibration::RTp
         */
        cv::Mat RTo;

        /**
         * Mean re-projection error
         * (root mean square)
         */
        double error;
    } CrossCalibration;

    void write_calibration(const std::string &file, const Calibration &c);

    Calibration read_calibration(const std::string &file);

    void write_cross_calibration(const std::string &file, const CrossCalibration &c);

    CrossCalibration read_cross_calibration(const std::string &file);

    std::string utc_iso_date_str_now();
}

#endif //XMOTION_JSON_OCV_H
