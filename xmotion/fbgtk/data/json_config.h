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

        /**
         * Single camera calibration.
         * Produces camera intrinsic properties (K, D).
         * Json value: "calibration"
         */
        CALIBRATION,

        /**
         * Chain calibration.
         * Produces chain of stereo calibrated pairs (RT, E, F).
         * Json value: "chain_calibration"
         */
        CHAIN_CALIBRATION,

        /**
         * Cross calibration (similar to chain_calibration)
         * Json value: "cross_calibration"
         */
        CROSS_CALIBRATION,

        /**
         * Compose calibration chain into single pair.
         * ie.: ( (0,1), (1,2), (2,3) ) -> (0,3).
         * Json value: "compose"
         */
        COMPOSE,

        /**
         * Motion capture pose estimation.
         * Produces stream of 3D points in world space coordinates of human body.
         * Json value: "pose"
         */
        POSE
    };

    typedef struct {
        /**
         * Capture device id.
         * i.e: "/dev/video1" or "usb-0000:02:00.0-2" etc.
         */
        std::string id;

        /**
         * Capture device name (arbitrary)
         */
        std::string name;

        /**
         * Capture codec, ie: "MJPG" / "YUYV" / "BGR3" / "H264" / etc
         */
        std::string codec;

        /**
         * Captured frame width
         */
        int width;

        /**
         * Captured frame height
         */
        int height;

        /**
         * Capture buffer size.
         * May introduce lag, but smooths fps
         */
        int buffer;

        /**
         * Capture FPS
         */
        int fps;

        /**
         * Optional, frame flip (horizontal/vertical/both).
         * Applied before (optional) rotation!
         */
        Flip flip;

        /**
         * Optional frame subregion
         */
        Region region;

        /**
         * Optional frame rotation.
         * Applied after (optional) flipping!
         */
        bool rotate;

        /**
         * Optional array of filters
         */
        std::vector<Filter> filters;
    } Capture;

    typedef struct {
        /**
         * Array of capture devices (often cameras)
         */
        std::vector<PoseDevice> devices;

        /**
         * Chain calibration config
         */
        ChainCalibration chain;

        /**
         * Cross calibration config
         */
        CrossCalibration cross;

        /**
         * Show epipolar lines (debug)
         */
        bool show_epilines;

        /**
         * Perform segmentation.
         * Used in helper background subtraction heuristics
         */
        bool segmentation;

        /**
         * Optional, number of dedicated cpu threads
         */
        int threads;
    } Pose;

    typedef struct {
        /**
         * Calibration compose output (result file) name
         */
        std::string name;

        /**
         * Chain calibration config
         */
        FileNames chain;
    } Compose;

    /**
     * THIS IS TOP LEVEL CONFIG, whom json config file is built on
     */
    typedef struct {
        /**
         * Configuration type
         */
        ConfigType type;

        /**
         * Miscellaneous configuration
         */
        Misc misc;

        /**
         * Gui configuration
         */
        Gui gui;

        /**
         * Motion capture pose configuration.
         * Used when type: "pose"
         */
        Pose pose;

        /**
         * Array of capturing devices
         */
        std::vector<Capture> captures;

        /**
         * Calibration configuration.
         * Used when type: "calibration" / "chain_calibration" / "cross_calibration"
         */
        Calibration calibration;

        /**
         * Calibration compose configuration.
         * Used when type: "compose"
         */
        Compose compose;
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
