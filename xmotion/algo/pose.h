
//
// Created by henryco on 4/22/24.
//
#pragma clang diagnostic push
#pragma ide diagnostic ignored "modernize-use-nodiscard"

#ifndef XMOTION_TRIANGULATION_H
#define XMOTION_TRIANGULATION_H

#include "i_logic.h"
#include "../utils/thread_pool.h"
#include "../dnn/pose_pipeline.h"

namespace xm::nview {

    using DetectorModel = eox::dnn::box::Model;
    using BodyModel = eox::dnn::pose::Model;

    typedef struct Initial {
        DetectorModel detector_model = eox::dnn::box::F_16;
        BodyModel body_model = eox::dnn::pose::FULL_F32;

        float roi_center_window = 0.f;
        float roi_clamp_window = 0.f;
        float roi_margin = 0.f;
        float roi_scale = 1.2f;
        float roi_padding_x = 0.f;
        float roi_padding_y = 0.f;
        float threshold_detector = 0.5f;
        float threshold_presence = 0.5f;
        float threshold_pose = 0.5f;
        float filter_velocity_factor = 0.5;
        int filter_windows_size = 30;
        int filter_target_fps = 30;

        bool segmentation;
        int threads;
        int views;
    } Initial;

    typedef struct Result {
        bool error;
    } Result;
}

namespace xm {

    class Pose : public xm::Logic {
    private:
        std::vector<cv::Mat> images{};
        xm::nview::Result results{};
        xm::nview::Initial config{};

        std::vector<std::unique_ptr<eox::util::ThreadPool>> workers;
        std::vector<std::unique_ptr<eox::dnn::PosePipeline>> poses;

        bool active = false;
        bool DEBUG = false;

    public:
        Pose() = default;

        Pose(Pose &&ref) = default;

        Pose(Pose &src) = delete;

        ~Pose() override;

        void init(const xm::nview::Initial &params);

        Pose &proceed(float delta, const std::vector<cv::Mat> &frames) override;

        bool is_active() const override;

        void start() override;

        void stop() override;

        const std::vector<cv::Mat> &frames() const override;

        void debug(bool _debug) override;

        const xm::nview::Result &result() const;

    protected:
        void release();

        void enqueue_inference(std::vector<std::future<eox::dnn::PosePipelineOutput>> &io_features,
                               const std::vector<cv::Mat> & in_frames,
                               std::vector<cv::Mat> & out_frames
        );

        static bool resolve_inference(std::vector<std::future<eox::dnn::PosePipelineOutput>> &in_features,
                                      std::vector<eox::dnn::PosePipelineOutput> &out_results);
    };

} // xm

#endif //XMOTION_TRIANGULATION_H

#pragma clang diagnostic pop