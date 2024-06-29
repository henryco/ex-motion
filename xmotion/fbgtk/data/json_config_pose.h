//
// Created by henryco on 28/06/24.
//

#ifndef XMOTION_JSON_CONFIG_POSE_H
#define XMOTION_JSON_CONFIG_POSE_H

namespace xm::data {

    namespace pose {
        enum ModelBody {
            HEAVY_ORIGIN = 0,
            FULL_ORIGIN = 1,
            LITE_ORIGIN = 2,

            HEAVY_F32 = 3,
            FULL_F32 = 4,
            LITE_F32 = 5,

            HEAVY_F16 = 6,
            FULL_F16 = 7,
            LITE_F16 = 8
        };

        enum ModeDetector {
            ORIGIN = 0,
            F_32 = 1,
            F_16 = 2
        };
    }

    typedef struct {
        /**
         * Distance between detectors and actual ROI middle point
         * for which detected ROI should be rolled back to previous one
         *
         * [0.0 ... 1.0]
         */
        float rollback_window;

        /**
         * Distance between actual and predicted ROI middle point
         * for which should stay unchanged (helps reducing jittering)
         *
         * [0.0 ... 1.0]
         */
        float center_window;

        /**
        * Acceptable ratio of clamped to original ROI size.
        * Zero (0) means every size is acceptable, One (1)
        * means only original (non-clamped) ROI is acceptable.
        *
        * [0.0 ... 1.0]
        */
        float clamp_window;

        /**
         * Scaling factor for ROI (multiplication)
         */
        float scale;

        /**
         * Margins added to ROI
         */
        float margin;

        /**
         * Horizontal paddings added to ROI
         */
        float padding_x;

        /**
        * Vertical paddings added to ROI
        */
        float padding_y;
    } PoseRoi;

    typedef struct {
        /**
        * Threshold score for detector ROI presence
        *
        * [0.0 ... 1.0]
        */
        float detector;

        /**
        * Threshold score for landmarks presence
        *
        * [0.0 ... 1.0]
        */
        float marks;

        /**
         * Threshold score for pose presence
         *
         * [0.0 ... 1.0]
         */
        float pose;

        /**
        * Threshold score for detector ROI distance to body marks. \n
        * In other works: How far marks should be from detectors ROI borders. \n
        * Currently implemented only for horizontal axis.
        *
        * [0.0 ... 1.0]
        */
        float roi;
    } PoseThresholds;

    typedef struct {
        /**
        * Low-pass filter velocity scale: lower -> smoother, but adds lag.
        */
        float velocity;

        /**
         * Low-pass filter window size: higher -> smoother, but adds lag.
         */
        int window;

        /**
         * Low-pass filter target fps.
         * Important to properly calculate points movement speed.
         */
        int fps;
    } PoseFilter;

    typedef struct {
        /**
         * BlazePose detector model
         */
        pose::ModeDetector detector;

        /**
         * BlazePose body model
         */
        pose::ModelBody body;
    } PoseModel;

    typedef struct {
        /**
         * Undistort input image
         */
        bool source;

        /**
         * Undistort position of localized points
         */
        bool points;

        /**
         * [0.0 ... 1.0]
         * Free scaling parameter:
         * 0 - only valid pixels
         * 1 - all pixels
         */
        float alpha;
    } PoseUndistort;

    typedef struct {
        /**
         * Name of the file with device intrinsic parameters
         * (camera calibration: K, D, etc)
         */
        std::string intrinsics;
        PoseThresholds threshold;
        PoseUndistort undistort;
        PoseFilter filter;
        PoseModel model;
        PoseRoi roi;
    } PoseDevice;

}

#endif //XMOTION_JSON_CONFIG_POSE_H
