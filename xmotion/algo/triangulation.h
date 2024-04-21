
//
// Created by henryco on 4/22/24.
//
#pragma clang diagnostic push
#pragma ide diagnostic ignored "modernize-use-nodiscard"

#ifndef XMOTION_TRIANGULATION_H
#define XMOTION_TRIANGULATION_H

#include "i_logic.h"

namespace xm::nview {
    typedef struct {
        // TODO
    } Result;
}

namespace xm {

    class Triangulation : public xm::Logic<xm::nview::Result> {

    public:
        xm::nview::Result proceed(const std::vector<cv::Mat> &frames) override;

        std::vector<cv::Mat> frames() const override;

        void start() override;

        void stop() override;
    };

} // xm

#endif //XMOTION_TRIANGULATION_H

#pragma clang diagnostic pop