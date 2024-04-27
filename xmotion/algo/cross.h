#pragma clang diagnostic push
#pragma ide diagnostic ignored "modernize-use-nodiscard"
//
// Created by henryco on 4/27/24.
//

#ifndef XMOTION_CROSS_H
#define XMOTION_CROSS_H

#include "i_logic.h"
#include "../utils/timer.h"

namespace xm::cross {
    typedef struct Result {
        int remains_cap;
        int remains_ms;
        bool ready;

        /**
         * Rotation matrix 3x3
         */
        std::vector<cv::Mat> R;

        /**
         * Translation vector
         */
        std::vector<cv::Mat> T;

        /**
         * Essential matrix
         */
        std::vector<cv::Mat> E;

        /**
         * Fundamental matrix
         */
        std::vector<cv::Mat> F;

        /**
         * Mean re-projection error
         * (root mean square)
         */
        std::vector<double> mre;

        /**
         * Number of cross calibrated pairs
         */
        int total;
    } Result;

    typedef struct Initial {
        int delay = 1000;
        int total = 10;
        int columns = 9;
        int rows = 7;
        float size = 30;

        int views = 2;
        std::vector<cv::Mat> K;
        std::vector<cv::Mat> D;
    } Initial;
}

namespace xm {

    class CrossCalibration : public xm::Logic {

    private:
        std::vector<std::vector<std::vector<cv::Point2f>>> image_points{};

        std::vector<cv::Mat> images{};
        xm::cross::Result results{};
        xm::cross::Initial config;
        eox::utils::Timer timer{};

        bool active = false;
        bool DEBUG = false;

    public:
        void init(const xm::cross::Initial &params);

        CrossCalibration &proceed(float delta, const std::vector<cv::Mat> &frames) override;

        bool is_active() const override;

        void start() override;

        void stop() override;

        const std::vector<cv::Mat> &frames() const override;

        void debug(bool _debug) override;

        const xm::cross::Result &result() const;

    protected:
        bool capture_squares(const std::vector<cv::Mat> &_frames);

        void calibrate();

        void put_debug_text();
    };
}

#endif //XMOTION_CROSS_H

#pragma clang diagnostic pop