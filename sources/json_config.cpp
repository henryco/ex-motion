//
// Created by henryco on 4/18/24.
//

#include "../xmotion/data/json_config.h"
#include <fstream>

namespace xm::data {
    NLOHMANN_JSON_SERIALIZE_ENUM(ConfigType, {
        {INVALID, nullptr},
        {CALIBRATION, "calibration"},
    })

    void from_json(const nlohmann::json &j, Camera &c) {
        j.at("id").get_to(c.id);
        j.at("name").get_to(c.name);
        j.at("width").get_to(c.width);
        j.at("height").get_to(c.height);

        c.codec = j.value("codec", "MJPG");
        c.buffer = j.value("buffer", 2);
        c.fast = j.value("fast", false);
        c.fps = j.value("fps", 30);
    }

    void from_json(const nlohmann::json &j, Gui &g) {
        g.scale = j.value("scale", 1.f);
    }

    void from_json(const nlohmann::json &j, Pattern &p) {
        j.at("columns").get_to(p.columns);
        j.at("rows").get_to(p.rows);
        j.at("size").get_to(p.size);
    }

    void from_json(const nlohmann::json &j, Calibration &c) {
        j.at("pattern").get_to(c.pattern);
        c.delay = j.value("delay", 5000);
        c.total = j.value("total", 10);
    }

    void from_json(const nlohmann::json &j, JsonConfig &c) {
        j.at("type").get_to(c.type);
        j.at("camera").get_to(c.camera);

        c.camera_names = {};
        for (const auto &item: c.camera) {
            c.camera_names.push_back(item.name + " [ " + item.id + " ]");
        }

        c.gui = j.value("gui", (Gui) {
                .scale = 1.f
        });

        if (c.type == ConfigType::CALIBRATION) {
            j.at("calibration").get_to(c.calibration);
        }
    }

    JsonConfig config_from_file(const std::string &path) {
        nlohmann::json content;

        std::ifstream file(path);
        if (!file.is_open())
            throw std::runtime_error("Cannot open file: " + path);

        file >> content;
        file.close();

        return content.template get<JsonConfig>();
    }

    std::string prepare_project_file(const std::string &path) {
        std::string project_path;
        if (std::filesystem::exists(path) && std::filesystem::is_directory(path)) {
            std::filesystem::path root = path;
            std::filesystem::path file = "config.json";
            project_path = (root / file).string();
        }

        if (!std::filesystem::exists(project_path))
            throw std::runtime_error("Cannot locate: " + project_path);

        return project_path;
    }

    std::string prepare_project_file(const char *c_str) {
        return prepare_project_file(std::string(c_str));
    }
}