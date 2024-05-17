//
// Created by henryco on 4/18/24.
//

#ifndef XMOTION_JSON_CONFIG_H
#define XMOTION_JSON_CONFIG_H

#include <string>
#include <vector>
#include "nlohmann/json.hpp"

namespace xm::data {

    enum ConfigType {
        INVALID = -1,
        CALIBRATION,
        CROSS_CALIBRATION,
        POSE
    };

    namespace board {
        enum Type {
            PLAIN = -1,
            CHESSBOARD,
            RADON
        };
    }

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
        int x;
        int y;
        int w;
        int h;
    } Region;

    typedef struct {
        bool x;
        bool y;
    } Flip;

    typedef struct {
        std::string id;
        std::string name;
        std::string codec;
        int width;
        int height;
        int buffer;
        int fps;
        Flip flip;
        Region region;
        bool rotate;
    } Capture;

    typedef struct {
        std::vector<Capture> capture;
        bool fast;
        bool dummy;

        std::vector<std::string> _names;
        std::vector<std::string> _ids;
    } Camera;

    typedef struct {
        std::vector<int> layout;
        bool vertical;
        float scale;
        int fps;
    } Gui;

    typedef struct {
        board::Type type;
        int columns;
        int rows;
        float size;
    } Pattern;

    typedef struct {
        float x;
        float y;
        bool fix;
    } Intrinsic;

    typedef struct {
        Intrinsic f;
        Intrinsic c;
    } Intrinsics;

    typedef struct {
        std::vector<std::string> calibrated;
        bool closed;
    } Cross;

    typedef struct {
        std::string name;
        Intrinsics intrinsics;
        Pattern pattern;
        Cross cross;
        int total;
        int delay;
    } Calibration;

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
        PoseThresholds threshold;
        PoseFilter filter;
        PoseRoi roi;
        // TODO
    } PoseDevice;

    typedef struct {
        std::vector<PoseDevice> devices;
        bool segmentation;
        int threads;
    } Pose;

    typedef struct {
        bool debug;
        int cpu;
    } Misc;

    typedef struct {
        ConfigType type;
        Misc misc;
        Gui gui;
        Pose pose;
        Camera camera;
        Calibration calibration;
    } JsonConfig;

    JsonConfig config_from_file(const std::string &file);

    std::string prepare_project_file(const std::string &path);

    std::string prepare_project_file(const char *c_str);

    std::string prepare_project_dir(const std::string &path);

    std::string prepare_project_dir(const char *c_str);

} // xm

#endif //XMOTION_JSON_CONFIG_H
