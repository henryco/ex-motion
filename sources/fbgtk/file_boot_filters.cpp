//
// Created by henryco on 6/2/24.
//

#include "../../xmotion/fbgtk/file_boot.h"
#include "../../xmotion/core/filter/chroma_key.h"
#include "../../xmotion/core/utils/cv_utils.h"

namespace xm {

    void FileBoot::filter_frames(std::vector<cv::UMat> &frames) {
        const auto t0 = std::chrono::system_clock::now();

        std::vector<std::future<cv::UMat>> futures;
        for (auto &frame: frames) {
            for (const auto &filter: filters) {
                frame = filter->filter(frame);
//
//                futures.push_back(thread_pool->execute<cv::UMat>([&frame, &filter]() -> cv::UMat {
//                    return filter->filter(frame);
//                }));

            }
        }

        for (auto &future: futures) {
            auto r = future.get();
        }

        const auto t1 = std::chrono::system_clock::now();
        const auto d = duration_cast<std::chrono::nanoseconds>((t1 - t0)).count();
        log->info("time: {}", d);
    }

    void FileBoot::prepare_filters() {
        if (!config.filters._present)
            return;

        // background filter
        opt_filter_chroma();
        opt_filter_delta();
    }

    void FileBoot::opt_filter_chroma() {
        if (!config.filters.background.chroma._present)
            return;

        const auto &conf = config.filters.background.chroma;
        auto filter = std::make_unique<xm::chroma::ChromaKey>();
        filter->init({
            .range = cv::Scalar(conf.range.h, conf.range.s, conf.range.l),
            .color = xm::ocv::parse_hex_to_bgr(conf.replace),
            .key = xm::ocv::parse_hex_to_bgr(conf.key),
            .refine = conf.refine,
            .fine = conf.fine,
            .blur = conf.blur,
            .power = conf.power,
        });

        filters.push_back(std::move(filter));
    }

    void FileBoot::opt_filter_delta() {
        if (!config.filters.background.delta._present)
            return;
        // TODO
    }
}