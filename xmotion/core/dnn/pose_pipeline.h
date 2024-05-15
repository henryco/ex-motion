//
// Created by henryco on 1/10/24.
//

#ifndef STEREOX_POSE_PIPELINE_H
#define STEREOX_POSE_PIPELINE_H

#include <vector>
#include <spdlog/logger.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include "../utils/velocity_filter.h"
#include "net/pose_detector.h"
#include "net/blaze_pose.h"
#include "net/pose_roi.h"

namespace eox::dnn {

    using PosePipelineOutput = struct {

        /**
         * pose landmarks in frame's coordinate system
         */
        eox::dnn::Landmark landmarks[39];

        /**
         * pose landmarks in world space
         */
        eox::dnn::Coord3d ws_landmarks[39];

        /**
         * segmentation array
         */
        float segmentation[256 * 256];

        /**
         * presence flag
         */
        bool present;

        /**
         * presence score
         */
        float score;
    };

    class PosePipeline {

        static inline const auto log =
                spdlog::stdout_color_mt("pose_pipeline");

    private:
        std::vector<eox::sig::VelocityFilter> filters;
        eox::dnn::PoseRoi roiPredictor;
        eox::dnn::PoseDetector detector;
        eox::dnn::BlazePose pose;

        bool preserved_roi = false;
        bool discarded_roi = false;
        bool rollback_roi = false;
        bool prediction = false;
        bool initialized = false;

        /**
         * Region of Interest
         */
        eox::dnn::RoI roi;

        /**
         * Margins added to ROI
         */
        float roi_margin = 0.f;

        /**
         * Horizontal paddings added to ROI
         */
        float roi_padding_x = 0.f;

        /**
         * Vertical paddings added to ROI
         */
        float roi_padding_y = 0.f;

        /**
         * Scaling factor for ROI (multiplication)
         */
        float roi_scale = 1.2f;

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
         * Acceptable ratio of clamped to original ROI size. \n
         * Zero (0) means every size is acceptable \n
         * One (1) means only original (non-clamped) ROI is acceptable \n
         *
         * [0.0 ... 1.0]
         */
        float roi_clamp_window = 0.f;

        /**
         * Threshold score for landmarks presence
         *
         * [0.0 ... 1.0]
         */
        float threshold_marks = 0.5;

        /**
         * Threshold score for detector ROI presence
         *
         * [0.0 ... 1.0]
         */
        float threshold_detector = 0.5;

        /**
         * Threshold score for body presence
         *
         * [0.0 ... 1.0]
         */
        float threshold_pose = 0.5;

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
        float f_v_scale = 0.5;

        /**
         * Low-pass filter window size: higher -> smoother, but adds lag.
         */
        int f_win_size = 30;

        /**
         * Low-pass filter target fps.
         * Important to properly calculate points movement speed.
         */
        int f_fps = 30;

        // DEBUG VARIABLES
        float _detector_score = 0;
        float _pose_score = 0;
        float _roi_score = 0;

    public:
        void init();

        PosePipelineOutput pass(const cv::Mat &frame);

        PosePipelineOutput pass(const cv::Mat &frame, cv::Mat &segmented);

        PosePipelineOutput pass(const cv::Mat &frame, cv::Mat &segmented, cv::Mat &debug);

        void setBodyModel(eox::dnn::pose::Model model);

        void setDetectorModel(eox::dnn::box::Model model);

        void enableSegmentation(bool enable);

        void setMarksThreshold(float threshold);

        void setPoseThreshold(float threshold);

        void setDetectorThreshold(float threshold);

        void setFilterWindowSize(int size);

        void setFilterVelocityScale(float scale);

        void setFilterTargetFps(int fps);

        void setRoiPredictionWindow(float window);

        void setRoiClampWindow(float window);

        void setRoiRollbackWindow(float window);

        void setRoiMargin(float margin);

        void setRoiPaddingX(float padding);

        void setRoiPaddingY(float padding);

        void setRoiScale(float scale);

        void setRoiThreshold(float threshold);

        [[nodiscard]] float getRoiThreshold() const;

        [[nodiscard]] float getRoiRollbackWindow() const;

        [[nodiscard]] float getRoiScale() const;

        [[nodiscard]] float getRoiMargin() const;

        [[nodiscard]] float getRoiPaddingX() const;

        [[nodiscard]] float getRoiPaddingY() const;

        [[nodiscard]] float getRoiClampWindow() const;

        [[nodiscard]] float getRoiPredictionWindow() const;

        [[nodiscard]] float getFilterVelocityScale() const;

        [[nodiscard]] int getFilterWindowSize() const;

        [[nodiscard]] int getFilterTargetFps() const;

        [[nodiscard]] float getPoseThreshold() const;

        [[nodiscard]] float getDetectorThreshold() const;

        [[nodiscard]] float getMarksThreshold() const;

        [[nodiscard]] bool segmentation() const;

        [[nodiscard]] eox::dnn::pose::Model bodyModel() const;

        [[nodiscard]] eox::dnn::box::Model getDetectorModel() const;

    protected:
        [[nodiscard]] PosePipelineOutput inference(const cv::Mat &frame, cv::Mat &segmented, cv::Mat *debug);

        void performSegmentation(float segmentation_array[128 * 128], const cv::Mat &frame, cv::Mat &out) const;

        void drawJoints(const eox::dnn::Landmark landmarks[39], cv::Mat &output) const;

        void drawLandmarks(const eox::dnn::Landmark landmarks[39], const eox::dnn::Coord3d ws3d[39], cv::Mat &output) const;

        void drawRoi(cv::Mat &output) const;

        void printMetadata(cv::Mat &output) const;

        [[nodiscard]] std::chrono::nanoseconds timestamp() const;
    };

} // eox

#endif //STEREOX_POSE_PIPELINE_H
