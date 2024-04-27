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
         * Mean re-projection error
         * (root mean square)
         */
        double mre_1;

        /**
         * Same as mre_1, calculated explicitly
         */
        double mre_2;

    } Result;

    typedef struct Initial {
        int delay = 1000;
        int total = 10;
        int columns = 9;
        int rows = 7;
        float size = 30;
    } Initial;
}

namespace xm {

    class CrossCalibration : public xm::Logic {

    private:
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