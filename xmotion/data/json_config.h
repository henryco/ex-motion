//
// Created by henryco on 4/18/24.
//

#ifndef XMOTION_JSON_CONFIG_H
#define XMOTION_JSON_CONFIG_H

#include <string>
#include <vector>
#include <nlohmann/json.hpp>

namespace xm::data {

    class Camera {
    public:
        std::string id;
        std::string name;
        std::string codec;
        int width{};
        int height{};
        int buffer{};
        int fps{};
        bool fast{};

        NLOHMANN_DEFINE_TYPE_INTRUSIVE(Camera, id, name, codec, width, height, buffer, fps, fast)
    };

    class Gui {
    public:
        float scale;

        NLOHMANN_DEFINE_TYPE_INTRUSIVE(Gui, scale)
    };

    class Pattern {
    public:
        int columns{};
        int rows{};
        float size{};

        NLOHMANN_DEFINE_TYPE_INTRUSIVE(Pattern, columns, rows, size)
    };

    class Calibration {
    public:
        Pattern pattern;
        int total{};
        int delay{};

        NLOHMANN_DEFINE_TYPE_INTRUSIVE(Calibration, pattern, total, delay)
    };

    class JsonConfig {
    public:
        std::string type;
        std::vector<Camera> camera;
        Calibration calibration;
        Gui gui;

        NLOHMANN_DEFINE_TYPE_INTRUSIVE(JsonConfig, type, camera, calibration, gui)
    };

    JsonConfig config_from_file(const std::string &file);

    std::string prepare_project_file(const std::string &path);

    std::string prepare_project_file(const char *c_str);

} // xm

#endif //XMOTION_JSON_CONFIG_H
