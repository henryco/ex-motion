//
// Created by henryco on 6/2/24.
//

#include "../../xmotion/fbgtk/file_worker.h"
#include "../../xmotion/core/filter/chroma_key.h"
#include "../../xmotion/core/utils/cv_utils.h"
#include "../../xmotion/core/filter/bg_subtract.h"
#include "../../xmotion/core/filter/blur.h"

namespace xm {

    void FileWorker::filter_frames(std::vector<xm::ocl::Image2D> &frames) {
        const auto t0 = std::chrono::system_clock::now();

        std::vector<xm::ocl::iop::ClImagePromise> frames_p_vec;
        frames_p_vec.reserve(frames.size());
        for (const auto &frame: frames)
            frames_p_vec.push_back(frame);

        int i = 0; for (auto &frame: frames_p_vec) {

            for (auto &filter: filters.at(i)) {

                if (!do_filter) {
                    filter->reset();
                    filter->stop();
                } else {
                    filter->start();
                }

                frame = filter->filter(frame);
            }

            i++;
        }

        xm::ocl::iop::ClImagePromise::finalizeAll(frames_p_vec);
        std::vector<xm::ocl::Image2D> out;
        out.reserve(frames_p_vec.size());
        for (auto &frame_p: frames_p_vec)
            out.push_back(frame_p.getImage2D());

        frames = out;

        const auto t1 = std::chrono::system_clock::now();
        const auto d = duration_cast<std::chrono::nanoseconds>((t1 - t0)).count();
        log->info("time: {}", d);
    }

    void FileWorker::prepare_filters() {
        filters.clear();
        filters.reserve(config.camera.capture.size());
        for (const auto &capture: config.camera.capture) {
            std::vector<std::unique_ptr<xm::Filter>> vec;

            for (const auto &f: capture.filters) {
                if (f.blur._present) {
                    auto filter = std::make_unique<xm::filters::Blur>();
                    filter->init(f.blur.power);
                    vec.push_back(std::move(filter));
                    continue;
                }

                if (f.chroma._present) {
                    const auto &conf = f.chroma;
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
                    vec.push_back(std::move(filter));
                    continue;
                }

                if (f.difference._present) {
                    auto filter = std::make_unique<xm::filters::BgLbpSubtract>();

                    vec.push_back(std::move(filter));
                    continue;
                }
            }

            filters.push_back(std::move(vec));
        }
    }
}