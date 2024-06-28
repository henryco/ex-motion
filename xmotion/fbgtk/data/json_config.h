//
// Created by henryco on 4/18/24.
//

#ifndef XMOTION_JSON_CONFIG_H
#define XMOTION_JSON_CONFIG_H

#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include "json_config_calibration.h"
#include "json_config_filters.h"
#include "json_config_common.h"
#include "json_config_pose.h"
#include "json_config_gui.h"

namespace xm::data {

    enum ConfigType {
        INVALID = -1,
        CALIBRATION,
        CHAIN_CALIBRATION,
        CROSS_CALIBRATION,
        COMPOSE,
        POSE
    };

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
        std::vector<PoseDevice> devices;
        ChainCalibration chain;
        CrossCalibration cross;
        bool show_epilines;
        bool segmentation;
        int threads;
    } Pose;

    typedef struct {
        std::string replace; // hex color
        int delay;
        bool _present;
    } Delta;

    typedef struct {
        Chroma chroma;
        Delta delta;
        bool _present;
    } Background;

    typedef struct {
        Background background;
        bool _present;
    } Filters;

    typedef struct {
        ConfigType type;
        Filters filters;
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

    std::filesystem::path create_dir_rec(const std::filesystem::path &path);

    std::vector<std::string> list_files(const std::filesystem::path &file);

    bool numeric_comparator_asc(const std::string &a, const std::string &b);
} // xm

#endif //XMOTION_JSON_CONFIG_H
