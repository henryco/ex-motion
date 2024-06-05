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

    typedef struct Result {
        int remains_cap;
        int remains_ms;
        bool ready;

        /**
         * Calibration matrix 3x3
         * \code
         * ┌ax    xo┐
         * │   ay yo│
         * └       1┘
         * \endcode
         */
        cv::Mat K;

        /**
         * Distortion coefficients
         */
        cv::Mat D;

        /**
         * Mean re-projection error
         * (root mean square)
         */
        double mre_1;

        /**
         * Same as mre_1, calculated explicitly
         */
        double mre_2;

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

    } Result;

    typedef struct Initial {
        int delay = 1000;
        int total = 10;
        int columns = 9;
        int rows = 7;
        float size = 30;
        bool sb = false;
        int width = 0;
        int height = 0;

        float fx = -1;
        float fy = -1;
        float cx = -1;
        float cy = -1;
        bool fix_f = false;
        bool fix_c = false;
    } Initial;
}

namespace xm {

    class Calibration : public xm::Logic {

    private:
        std::vector<std::vector<cv::Point2f>> image_points{};
        std::vector<cv::UMat> images{};
        xm::calib::Result results{};
        xm::calib::Initial config;
        eox::utils::Timer timer{};

        bool active = false;
        bool DEBUG = false;

    public:
        void init(const xm::calib::Initial &params);

        Calibration &proceed(float delta, const std::vector<cv::UMat> &frames) override;

        bool is_active() const override;

        void start() override;

        void stop() override;

        const std::vector<cv::UMat> &frames() const override;

        void debug(bool _debug) override;

        const xm::calib::Result &result() const;

    protected:
        bool capture_squares(const cv::UMat &frame);

        void calibrate();

        void put_debug_text();
    };

} // xm

#endif //XMOTION_CALIBRATION_H

#pragma clang diagnostic pop