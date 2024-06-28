//
// Created by henryco on 6/2/24.
//

#include "../../xmotion/fbgtk/file_worker.h"
#include "../../xmotion/core/filter/chroma_key.h"
#include "../../xmotion/core/utils/cv_utils.h"
#include "../../xmotion/core/filter/bg_lbp_subtract.h"

namespace xm {

    void FileWorker::filter_frames(std::vector<xm::ocl::Image2D> &frames) {
        const auto t0 = std::chrono::system_clock::now();

        for (const auto &filter: filters) {
            if (!do_filter) {
                filter->reset();
                filter->stop();
                continue;
            } else {
                filter->start();
            }

            for (auto &frame: frames) {
                filter->filter(frame).waitFor().toImage2D(frame);
            }
        }

        const auto t1 = std::chrono::system_clock::now();
        const auto d = duration_cast<std::chrono::nanoseconds>((t1 - t0)).count();
        log->info("time: {}", d);
    }

    void FileWorker::prepare_filters() {
        if (!config.filters._present)
            return;

        // background filter
//        opt_filter_chroma();
        opt_filter_delta();
    }

    void FileWorker::opt_filter_chroma() {
        if (!config.filters.background.chroma._present)
            return;

        const auto &conf = config.filters.background.chroma;
        auto filter = std::make_unique<xm::filters::ChromaKey>();
        filter->init({
            .range = xm::ds::Color4u::hls((int) (conf.range.h * 255.f), (int) (conf.range.l * 255.f), (int) (conf.range.s * 255.f)),
            .color = xm::ocv::parse_hex_to_bgr_4u(conf.replace),
            .key = xm::ocv::parse_hex_to_bgr_4u(conf.key),
            .refine = conf.refine,
            .fine = conf.fine,
            .blur = conf.blur,
            .power = conf.power,
            .linear = conf.linear
        });

        filters.push_back(std::move(filter));
    }

    void FileWorker::opt_filter_delta() {
//        if (!config.filters.background.delta._present)
//            return;

        const auto &conf = config.filters.background.delta;
        auto filter = std::make_unique<xm::filters::BgLbpSubtract>();
        filter->init({
            // TODO
        });

        filters.push_back(std::move(filter));
    }
}