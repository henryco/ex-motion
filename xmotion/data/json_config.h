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
        std::string id;
        std::string name;
        std::string codec;
        int width;
        int height;
        int buffer;
        int fps;
        bool fast;
    } Camera;

    typedef struct {
        float scale;
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
        std::vector<Camera> camera;
        Calibration calibration;
        Gui gui;
    } JsonConfig;

    JsonConfig config_from_file(const std::string &file);

    std::string prepare_project_file(const std::string &path);

    std::string prepare_project_file(const char *c_str);

} // xm

#endif //XMOTION_JSON_CONFIG_H
