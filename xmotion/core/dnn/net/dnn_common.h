//
// Created by henryco on 1/4/24.
//

#ifndef STEREOX_DNN_COMMON_H
#define STEREOX_DNN_COMMON_H

#include <vector>
#include <opencv2/core/mat.hpp>

namespace eox::dnn {

    using Paddings = struct {
        float left;
        float right;
        float top;
        float bottom;
    };

    using Point = struct {
        float x, y;
    };

    using Box = struct {
        float x, y, w, h;
    };

    using RoI = struct {
        float x, y, w, h;

        /**
         * usually center, but sometimes just another point...
         */
        Point c;

        /**
         * usually end for radius, but sometimes just another point...
         */
        Point e;
    };

    using Coord3d = struct {
        /**
         * X
         */
        float x;

        /**
         * Y
         */
        float y;

        /**
         * Z
         */
        float z;
    };

    using Landmark = struct {

        /**
         * X
         */
        float x;

        /**
         * Y
         */
        float y;

        /**
         * Z
         */
        float z;

        /**
         * visibility (need to apply sigmoid)
         */
        float v;

        /**
         * presence (need to apply sigmoid)
         */
        float p;
    };

    using PoseOutput = struct {

        /**
         * 39x5 normalized (0,1) landmarks
         */
        eox::dnn::Landmark landmarks_norm[39];

        /*
         * 39x3 world space landmarks
         */
        eox::dnn::Coord3d landmarks_3d[39];

        /**
         * 1D 256x256 float32 array
         */
        float segmentation[256 * 256];

        /**
         * Probability [0,1]
         */
        float score;
    };

    using DetectedRegion = struct {

        /**
         * Detected SSD box
         */
        Box box;

        /**
         * Key point 0 - mid hip center
         * Key point 1 - point that encodes size & rotation (for full body)
         * Key point 2 - mid shoulder center
         * Key point 3 - point that encodes size & rotation (for upper body)
         */
        std::vector<Point> key_points;

        /**
         * Probability [0,1]
         */
        float score;

        /**
         * from -Pi to Pi radians
         */
        float rotation;

    };

    /**
     * see media/pose_landmark_topology.svg
     */
    const int body_joints[31][2] = {
            {0,  2},
            {0,  5},
            {2,  7},
            {5,  8},

            {10, 9},

            {12, 11},

            {12, 14},
            {14, 16},
            {16, 22},
            {16, 18},
            {16, 20},
            {18, 20},

            {11, 13},
            {13, 15},
            {15, 21},
            {15, 17},
            {15, 19},
            {17, 19},

            {12, 24},
            {24, 23},
            {11, 23},

            {24, 26},
            {26, 28},
            {28, 32},
            {28, 30},
            {32, 30},

            {23, 25},
            {25, 27},
            {27, 29},
            {27, 31},
            {29, 31},
    };

    namespace LM {
        const int NOSE = 0;

        const int SHOULDER = 11;
        const int SHOULDER_L = SHOULDER + 0;
        const int SHOULDER_R = SHOULDER + 1;

        const int HIP = 23;
        const int HIP_L = HIP + 0;
        const int HIP_R = HIP + 1;

        const int R = 33;
        const int R_MID = R + 0;
        const int R_END = R + 1;
    }

    double sigmoid(double x);

    Paddings get_letterbox_paddings(int width, int height, int size);

    Paddings get_letterbox_paddings(int width, int height, int bow_w, int box_h);

    cv::Mat convert_to_squared_blob(const cv::Mat &in, int size, bool keep_aspect_ratio = false);
    cv::UMat convert_to_squared_blob(const cv::UMat &in, int size, bool keep_aspect_ratio = false);

    cv::Mat convert_to_squared_blob(const cv::Mat &in, int width, int height, bool keep_aspect_ratio = false);
    cv::UMat convert_to_squared_blob(const cv::UMat &in, int width, int height, bool keep_aspect_ratio = false);

    cv::Mat remove_paddings(const cv::Mat &in, int width, int height);
    cv::UMat remove_paddings(const cv::UMat &in, int width, int height);

    RoI clamp_roi(const eox::dnn::RoI &roi, int width, int height);

    double normalize_radians(double angle);
}

#endif //STEREOX_DNN_COMMON_H