
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

#include <spdlog/logger.h>
#include <spdlog/sinks/stdout_color_sinks.h>

namespace xm::nview {

    using DetectorModel = eox::dnn::box::Model;
    using BodyModel = eox::dnn::pose::Model;

    typedef struct StereoPair {
        /**
         * Essential matrix
         */
        cv::Mat E;

        /**
         * Fundamental matrix
         */
        cv::Mat F;

        /**
         * Rotation-translation matrix
         * \code
         * ┌ R R R tx ┐
         * │ R R R ty │
         * │ R R R tz │
         * └ 0 0 0 1  ┘
         * \endcode
         */
        cv::Mat RT;

        /**
         * Rotation-translation matrix according to fist camera
         */
        cv::Mat RTo;
    } StereoPair;

    typedef struct Device {
        /**
         * BlazePose detector model
         */
        DetectorModel detector_model = eox::dnn::box::F_16;

        /**
         * BlazePose body model
         */
        BodyModel body_model = eox::dnn::pose::FULL_F32;

        /**
         * Distance between detectors and actual ROI mid point
         * for which detected ROI should be rolled back to previous one
         *
         * [0.0 ... 1.0]
         */
        float roi_rollback_window = 0.f;

        /**
         * Distance between actual and predicted ROI mid point
         * for which should stay unchanged (helps reducing jittering)
         *
         * [0.0 ... 1.0]
         */
        float roi_center_window = 0.f;

        /**
         * Acceptable ratio of clamped to original ROI size.
         * Zero (0) means every size is acceptable, One (1)
         * means only original (non-clamped) ROI is acceptable.
         *
         * [0.0 ... 1.0]
         */
        float roi_clamp_window = 0.f;

        /**
         * Margins added to ROI
         */
        float roi_margin = 0.f;

        /**
         * Scaling factor for ROI (multiplication)
         */
        float roi_scale = 1.2f;

        /**
         * Horizontal paddings added to ROI
         */
        float roi_padding_x = 0.f;

        /**
        * Vertical paddings added to ROI
        */
        float roi_padding_y = 0.f;

        /**
         * Threshold score for detector ROI presence
         *
         * [0.0 ... 1.0]
         */
        float threshold_detector = 0.5f;

        /**
         * Threshold score for landmarks presence
         *
         * [0.0 ... 1.0]
         */
        float threshold_marks = 0.5f;

        /**
         * Threshold score for pose presence
         *
         * [0.0 ... 1.0]
         */
        float threshold_pose = 0.5f;

        /**
         * Threshold score for detector ROI distance to body marks. \n
         * In other works: How far marks should be from detectors ROI borders. \n
         * Currently implemented only for horizontal axis.
         *
         * [0.0 ... 1.0]
         */
        float threshold_roi = 0.f;

        /**
         * Low-pass filter velocity scale: lower -> smoother, but adds lag.
         */
        float filter_velocity_factor = 0.5;

        /**
         * Low-pass filter window size: higher -> smoother, but adds lag.
         */
        int filter_windows_size = 30;

        /**
         * Low-pass filter target fps.
         * Important to properly calculate points movement speed.
         */
        int filter_target_fps = 30;

        /**
         * Undistort input image
         */
        bool undistort_source = false;

        /**
         * Undistort position of localized points
         */
        bool undistort_points = false;

        /**
         * [0.0 ... 1.0]
         * Free scaling parameter:
         * 0 - only valid pixels
         * 1 - all pixels
         */
        float undistort_alpha = 0;

        /**
           * Image width
           */
        int width;

        /**
         * Image height
         */
        int height;

        /**
        * Camera calibration matrix 3x3
        * \code
        * ┌ax    xo┐
        * │   ay yo│
        * └       1┘
        * \endcode
        */
        cv::Mat K;

        /**
         * Distortion coefficients
         */
        cv::Mat D;
    } Device;

    typedef struct Initial {

        /**
         * Camera devices
         */
        std::vector<Device> devices;

        /**
         * Stereo pairs
         */
        std::vector<StereoPair> pairs;

        /**
         * Enable pose segmentation
         */
        bool segmentation;

        /**
         * Number of worker threads
         */
        int threads;
    } Initial;

    typedef struct ReMaps {
        cv::Mat newK;
        cv::Mat map1;
        cv::Mat map2;
    } ReMaps;

    typedef struct Result {
        bool error;
        std::string err_msg;
    } Result;
}

namespace xm {

    class Pose : public xm::Logic {
        static inline const auto log =
                spdlog::stdout_color_mt("pose_estimation");

    private:
        /**
         * NxN matrix of stereo pairs (F, E and RT)
         * Pairs: (from -> to)
         * \code
         *    t0 t1 t2
         * f0 ┌X      ┐
         * f1 │   X   │
         * f2 └      X┘
         * \endcode
         */
        xm::nview::StereoPair **epipolar_matrix = nullptr;
        int epipolar_matrix_size = 0;

        std::vector<xm::nview::ReMaps> remap_maps{};
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

        cv::Mat undistorted(const cv::Mat &in, int index) const;

        std::vector<cv::Point2f> undistorted(const eox::dnn::Landmark *in, int num, int index) const;

        cv::Vec3f epi_line_from_point(const cv::Point2f &point, int idx_point, int idx_line) const;

        void points_from_epi_line(const cv::Mat &img, const cv::Vec3f &line, cv::Point2i &p1, cv::Point2i &p2) const;

        void init_epipolar_matrix();

        void init_undistort_maps();

        void init_validate();
    };

} // xm

#endif //XMOTION_TRIANGULATION_H

#pragma clang diagnostic pop