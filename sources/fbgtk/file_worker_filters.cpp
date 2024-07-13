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
//        const auto t0 = std::chrono::system_clock::now();

        const auto arr_size = frames.size();
        auto frames_p_arr = new xm::ocl::iop::ClImagePromise[arr_size];
        for (int i = 0; i < frames.size(); i++)
            frames_p_arr[i] = frames[i];

        for (int i = 0; i < arr_size; i++){
            auto &frame = frames_p_arr[i];

            for (auto &filter: filters.at(i)) {

                if (!do_filter) {
                    filter->reset();
                    filter->stop();
                } else {
                    filter->start();
                }

                frame = filter->filter(frame);
            }
        }

        xm::ocl::iop::ClImagePromise::finalizeAll(frames_p_arr, arr_size);
        std::vector<xm::ocl::Image2D> out;
        out.reserve(arr_size);
        for (int i = 0; i < arr_size; i++)
            out.push_back(frames_p_arr[i].getImage2D());
        delete[] frames_p_arr;

        frames = out;

//        const auto t1 = std::chrono::system_clock::now();
//        const auto d = duration_cast<std::chrono::nanoseconds>((t1 - t0)).count();
//        log->info("time: {}", d);
    }

    void FileWorker::prepare_filters() {
        filters.clear();
        filters.reserve(config.captures.size());
        for (const auto &capture: config.captures) {
            std::vector<std::unique_ptr<xm::Filter>> vec;

            for (const auto &f: capture.filters) {
                if (f.blur._present) {
                    auto filter = std::make_unique<xm::filters::Blur>();
                    filter->init(f.blur.blur);
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
                    const auto &conf = f.difference;
                    auto filter = std::make_unique<xm::filters::BgSubtract>();
                    filter->init({
                        .BASE_RESOLUTION = conf.BASE_RESOLUTION,
                        .color_channels = 3,
                        .adapt_on = conf.adapt_on,
                        .debug_on = conf.debug_on,
                        .morph_on = (conf.refine_gate + conf.refine_erode + conf.refine_dilate) > 0,
                        .ghost_on = conf.ghost_on,
                        .lbsp_on = conf.lbsp_on,
                        .norm_l2 = conf.norm_l2,
                        .mask_xc = false,
                        .linear = conf.linear,
                        .color_0 = conf.color_0,
                        .lbsp_0 = conf.lbsp_0,
                        .lbsp_d = conf.lbsp_d,
                        .n_matches = conf.n_matches,
                        .t_upper = conf.t_upper,
                        .t_lower = conf.t_lower,
                        .model_size = conf.model_size,
                        .ghost_l = conf.ghost_l,
                        .ghost_n = conf.ghost_n,
                        .ghost_n_inc = conf.ghost_n_inc,
                        .ghost_n_dec = conf.ghost_n_dec,
                        .alpha_d_min = conf.alpha_d_min,
                        .alpha_norm = conf.alpha_norm,
                        .ghost_t = conf.ghost_t,
                        .r_scale = conf.r_scale,
                        .r_cap = conf.r_cap,
                        .t_scale_inc = conf.t_scale_inc,
                        .t_scale_dec = conf.t_scale_dec,
                        .v_flicker_inc = conf.v_flicker_inc,
                        .v_flicker_dec = conf.v_flicker_dec,
                        .v_flicker_cap = conf.v_flicker_cap,
                        .kernel = static_cast<xm::filters::bgs::KernelType>((int) conf.kernel),
                        .color = xm::ocv::parse_hex_to_bgr_4u(conf.color),
                        .refine_gate = conf.refine_gate,
                        .refine_erode = conf.refine_erode,
                        .refine_dilate = conf.refine_dilate,
                        .refine_gate_threshold = conf.gate_threshold,
                        .gate_kernel = static_cast<xm::filters::bgs::KernelType>((int) conf.gate_kernel),
                        .erode_kernel = static_cast<xm::filters::bgs::KernelType>((int) conf.erode_kernel),
                        .dilate_kernel = static_cast<xm::filters::bgs::KernelType>((int) conf.dilate_kernel),
                    });
                    vec.push_back(std::move(filter));
                    continue;
                }
            }

            filters.push_back(std::move(vec));
        }
    }
}