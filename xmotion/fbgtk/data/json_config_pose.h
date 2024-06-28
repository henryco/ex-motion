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
        float rollback_window;
        float center_window;
        float clamp_window;
        float scale;
        float margin;
        float padding_x;
        float padding_y;
    } PoseRoi;

    typedef struct {
        float detector;
        float marks;
        float pose;
        float roi;
    } PoseThresholds;

    typedef struct {
        float velocity;
        int window;
        int fps;
    } PoseFilter;

    typedef struct {
        pose::ModeDetector detector;
        pose::ModelBody body;
    } PoseModel;

    typedef struct {
        bool source;
        bool points;
        float alpha;
    } PoseUndistort;

    typedef struct {
        std::string intrinsics;
        PoseThresholds threshold;
        PoseUndistort undistort;
        PoseFilter filter;
        PoseModel model;
        PoseRoi roi;
    } PoseDevice;

}

#endif //XMOTION_JSON_CONFIG_POSE_H
