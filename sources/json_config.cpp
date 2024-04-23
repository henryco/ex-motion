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

    void from_json(const nlohmann::json &j, Flip &f) {
        f.x = j.value("x", false);
        f.y = j.value("y", false);
    }

    void from_json(const nlohmann::json &j, Region &r) {
        j.at("w").get_to(r.w);
        j.at("h").get_to(r.h);
        r.x = j.value("x", 0);
        r.y = j.value("y", 0);
    }

    void from_json(const nlohmann::json &j, Intrinsics &t) {
        t.f_x = j.value("fx", -1.f);
        t.f_y = j.value("fy", -1.f);
        t.c_x = j.value("cx", -1.f);
        t.c_y = j.value("cy", -1.f);
    }

    void from_json(const nlohmann::json &j, Capture &c) {
        j.at("id").get_to(c.id);
        j.at("name").get_to(c.name);

        c.width = j.value("width", 0);
        c.height = j.value("height", 0);
        c.codec = j.value("codec", "MJPG");
        c.buffer = j.value("buffer", 2);
        c.fps = j.value("fps", 30);

        c.region = j.value("region", (Region) {
            .x = 0,
            .y = 0,
            .w = c.width,
            .h = c.height
        });

        c.flip = j.value("flip", (Flip) {
            .x = false,
            .y = false
        });

        c.intrinsics = j.value("intrinsics", (Intrinsics) {
            .f_x = -1,
            .f_y = -1,
            .c_x = -1,
            .c_y = -1
        });
    }

    void from_json(const nlohmann::json &j, Camera &c) {
        j.at("capture").get_to(c.capture);
        c.fast = j.value("fast", false);

        c._names = {};
        c._ids = {};
        for (const auto &d: c.capture) {
            c._names.push_back(d.name);
            c._ids.push_back(d.id);
        }
    }

    void from_json(const nlohmann::json &j, Gui &g) {
        g.vertical = j.value("vertical", false);
        g.scale = j.value("scale", 1.f);
        g.fps = j.value("fps", 144);
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
        c.fix = j.value("fix", false);
    }

    void from_json(const nlohmann::json &j, JsonConfig &c) {
        j.at("type").get_to(c.type);
        j.at("camera").get_to(c.camera);

        c.gui = j.value("gui", (Gui) {
                .vertical = false,
                .scale = 1.f,
                .fps = 300
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
        } else {
            project_path = path;
        }

        if (!std::filesystem::exists(project_path))
            throw std::runtime_error("Cannot locate: " + project_path);

        return project_path;
    }

    std::string prepare_project_file(const char *c_str) {
        return prepare_project_file(std::string(c_str));
    }

    std::string prepare_project_dir(const std::string &path) {
        std::filesystem::path file = path;
        if (is_directory(file))
            return path;
        return file.parent_path().string();
    }

    std::string prepare_project_dir(const char *c_str) {
        return prepare_project_dir(std::string(c_str));
    }
}