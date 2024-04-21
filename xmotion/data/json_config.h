//
// Created by henryco on 4/18/24.
//

#ifndef XMOTION_JSON_CONFIG_H
#define XMOTION_JSON_CONFIG_H

#include <string>
#include <vector>
#include <nlohmann/json.hpp>

namespace xm::data {

    enum ConfigType {
        INVALID = -1,
        CALIBRATION
    };

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
    } Capture;

    typedef struct {
        std::vector<Capture> capture;
        bool fast;

        std::vector<std::string> _names;
        std::vector<std::string> _ids;
    } Camera;

    typedef struct {
        bool vertical;
        float scale;
        int fps;
    } Gui;

    typedef struct {
        int columns;
        int rows;
        float size;
    } Pattern;

    typedef struct {
        Pattern pattern;
        int total;
        int delay;
    } Calibration;

    typedef struct {
        ConfigType type;
        Gui gui;
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
