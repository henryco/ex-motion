
//
// Created by henryco on 4/22/24.
//
#pragma clang diagnostic push
#pragma ide diagnostic ignored "modernize-use-nodiscard"

#ifndef XMOTION_TRIANGULATION_H
#define XMOTION_TRIANGULATION_H

#include "i_logic.h"

namespace xm::nview {
    typedef struct Result {
        // TODO
    } Result;

    typedef struct Initial {
        // TODO
        int threads;
        int views;
    } Initial;
}

namespace xm {

    class Triangulation : public xm::Logic {
    private:
        std::vector<cv::Mat> images{};
        xm::nview::Result results{};
        xm::nview::Initial config;

        bool active = false;
        bool DEBUG = false;

    public:
        void init(const xm::nview::Initial &params);

        Triangulation &proceed(float delta, const std::vector<cv::Mat> &frames) override;

        bool is_active() const override;

        void start() override;

        void stop() override;

        const std::vector<cv::Mat> &frames() const override;

        void debug(bool _debug) override;

        const xm::nview::Result &result() const;
    };

} // xm

#endif //XMOTION_TRIANGULATION_H

#pragma clang diagnostic pop