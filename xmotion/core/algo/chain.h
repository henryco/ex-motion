#pragma clang diagnostic push
#pragma ide diagnostic ignored "modernize-use-nodiscard"
//
// Created by henryco on 4/27/24.
//

#ifndef XMOTION_CROSS_H
#define XMOTION_CROSS_H

#include "i_logic.h"
#include "../utils/timer.h"

#include <spdlog/logger.h>
#include <spdlog/sinks/stdout_color_sinks.h>

namespace xm::chain {
    typedef struct Pair {
        /**
         * Rotation matrix 3x3
         */
        cv::Mat R;

        /**
         * Translation vector3 (x,y,z)
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
        * Mean re-projection error
        * (root mean square)
        */
        double mre;
    } Pair;

    typedef struct Result {
        int remains_cap;
        int remains_ms;
        bool ready;

        /**
         * Results for stereo calibrated pairs
         */
        std::vector<Pair> calibrated;

        /**
         * Number of total cross calibrated pairs
         */
        int total;

        /**
         * Current calibration pair (0-indexed)
         */
        int current;
    } Result;

    typedef struct Initial {
        int delay = 1000;
        int total = 10;
        int columns = 9;
        int rows = 7;
        float size = 30;
        bool sb = false;

        bool closed = false;
        int views = 2;
        std::vector<cv::Mat> K;
        std::vector<cv::Mat> D;
    } Initial;
}

namespace xm {

    class ChainCalibration : public xm::Logic {

        static inline const auto log =
                spdlog::stdout_color_mt("chain_calibration");

    private:
        // [pair_numb][image_number][2: (l,r)][points: Rows x Cols][2: (x,y)]
        std::vector<std::vector<std::vector<std::vector<cv::Point2f>>>> image_points{};

        std::vector<xm::ocl::Image2D> images{};
        xm::chain::Result results{};
        xm::chain::Initial config;
        eox::utils::Timer timer{};

        bool active = false;
        bool DEBUG = false;

        int counter = 0;
        int total_pairs = 0;
        int current_pair = 0;

    public:
        void init(const xm::chain::Initial &params);

        ChainCalibration &proceed(float delta, const std::vector<xm::ocl::Image2D> &frames) override;

        bool is_active() const override;

        void start() override;

        void stop() override;

        const std::vector<xm::ocl::Image2D> &frames() const override;

        void debug(bool _debug) override;

        const xm::chain::Result &result() const;

    protected:
        bool capture_squares(const std::vector<xm::ocl::Image2D> &_frames);

        void calibrate();

        void put_debug_text();
    };
}

#endif //XMOTION_CROSS_H

#pragma clang diagnostic pop