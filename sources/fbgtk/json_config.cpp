//
// Created by henryco on 4/18/24.
//

#include "../../xmotion/fbgtk/data/json_config.h"
#include <fstream>

namespace xm::data::def {
    Intrinsic intrinsic() {
        return {
                .x = -1,
                .y = -1,
                .fix = false
        };
    }

    Intrinsics intrinsics() {
        return {
                .f = intrinsic(),
                .c = intrinsic()
        };
    }

    Chain chain() {
        return {
                .intrinsics = std::vector<std::string>{},
                .closed = false
        };
    }

    GuiFrame guiFrame() {
        return {
          .w = 0,
          .h = 0
        };
    }

    Gui gui() {
        return {
                .layout = std::vector<int>{},
                .frame = guiFrame(),
                .vertical = false,
                .scale = 1.f,
                .fps = 300
        };
    }

    Misc misc() {
        return {
                .debug = false,
                .cpu = 8
        };
    }

    PoseRoi poseRoi() {
        return {
                .rollback_window = 0.f,
                .center_window = 0.f,
                .clamp_window = 0.f,
                .scale = 1.2f,
                .margin = 0.f,
                .padding_x = 0.f,
                .padding_y = 0.f
        };
    }

    PoseUndistort poseUndistort() {
        return {
          .source = false,
          .points = false,
          .alpha = 0.f
        };
    }

    PoseThresholds poseThresholds() {
        return {
                .detector = 0.5f,
                .marks = 0.5f,
                .pose = 0.5f,
                .roi = 0.f
        };
    }

    PoseFilter poseFilter() {
        return {
                .velocity = 0.5f,
                .window = 30,
                .fps = 30
        };
    }

    PoseModel poseModel() {
        return {
                .detector = pose::F_16,
                .body = pose::FULL_F32,
        };
    }

    PoseDevice poseDevice() {
        return {
                .calibration = "",
                .threshold = xm::data::def::poseThresholds(),
                .undistort = xm::data::def::poseUndistort(),
                .filter = xm::data::def::poseFilter(),
                .model = xm::data::def::poseModel(),
                .roi = xm::data::def::poseRoi()
        };
    }

    Pose pose() {
        return {
                .devices = {},
                .show_epilines = false,
                .segmentation = false,
                .threads = 0
        };
    }
}

namespace xm::data {
    NLOHMANN_JSON_SERIALIZE_ENUM(ConfigType, {
        { INVALID, nullptr },
        { CALIBRATION, "calibration" },
        { CROSS_CALIBRATION, "cross_calibration" },
        { CHAIN_CALIBRATION, "chain_calibration" },
        { POSE, "pose" },
    })

    namespace board {
        NLOHMANN_JSON_SERIALIZE_ENUM(Type, {
            { PLAIN, nullptr },
            { CHESSBOARD, "chessboard" },
            { RADON, "radon" }
        })
    }

    namespace pose {
        NLOHMANN_JSON_SERIALIZE_ENUM(ModelBody, {
            { FULL_F32, nullptr },

            { HEAVY_ORIGIN, "heavy" },
            { FULL_ORIGIN, "full" },
            { LITE_ORIGIN, "lite" },

            { HEAVY_F32, "heavy_f32" },
            { FULL_F32, "full_32" },
            { LITE_F32, "lite_f32" },

            { HEAVY_F16, "heavy_f16" },
            { FULL_F16, "full_f16" },
            { LITE_F16, "lite_f16" },
        })

        NLOHMANN_JSON_SERIALIZE_ENUM(ModeDetector, {
            { F_16, nullptr },
            { ORIGIN, "origin" },
            { F_32, "f_32" },
            { F_16, "f_16" },
        })
    }

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

    void from_json(const nlohmann::json &j, Intrinsic n) {
        n.fix = j.value("fix", false);
        n.x = j.value("x", -1.f);
        n.y = j.value("y", -1.f);
    }

    void from_json(const nlohmann::json &j, Intrinsics &t) {
        t.f = j.value("f", (Intrinsic) xm::data::def::intrinsic());
        t.c = j.value("c", (Intrinsic) xm::data::def::intrinsic());
    }

    void from_json(const nlohmann::json &j, Chain &c) {
        c.closed = j.value("closed", false);
        j.at("intrinsics").get_to(c.intrinsics);
    }

    void from_json(const nlohmann::json &j, Capture &c) {
        j.at("id").get_to(c.id);
        j.at("name").get_to(c.name);

        c.width = j.value("width", 0);
        c.height = j.value("height", 0);
        c.codec = j.value("codec", "MJPG");
        c.buffer = j.value("buffer", 2);
        c.fps = j.value("fps", 30);
        c.rotate = j.value("rotate", false);

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
    }

    void from_json(const nlohmann::json &j, Camera &c) {
        j.at("capture").get_to(c.capture);
        c.fast = j.value("fast", false);
        c.dummy = j.value("dummy", false);

        c._names = {};
        c._ids = {};
        for (const auto &d: c.capture) {
            c._names.push_back(d.name);
            c._ids.push_back(d.id);
        }
    }

    void from_json(const nlohmann::json &j, GuiFrame &f) {
        const auto def = xm::data::def::guiFrame();
        f.w = j.value("w", def.w);
        f.h = j.value("h", def.h);
    }

    void from_json(const nlohmann::json &j, Gui &g) {
        g.layout = j.value("layout", std::vector<int>{});
        g.frame = j.value("frame", xm::data::def::guiFrame());
        g.vertical = j.value("vertical", false);
        g.scale = j.value("scale", 1.f);
        g.fps = j.value("fps", 300);
    }

    void from_json(const nlohmann::json &j, Pattern &p) {
        p.type = j.value("type", board::Type::CHESSBOARD);
        j.at("columns").get_to(p.columns);
        j.at("rows").get_to(p.rows);
        j.at("size").get_to(p.size);
    }

    void from_json(const nlohmann::json &j, Calibration &c) {
        j.at("name").get_to(c.name);
        j.at("pattern").get_to(c.pattern);

        c.intrinsics = j.value("intrinsics", xm::data::def::intrinsics());
        c.chain = j.value("chain", xm::data::def::chain());
        c.delay = j.value("delay", 5000);
        c.total = j.value("total", 10);
    }

    void from_json(const nlohmann::json &j, PoseRoi &r) {
        const auto def = xm::data::def::poseRoi();
        r.padding_y = j.value("padding_y", def.padding_y);
        r.padding_x = j.value("padding_x", def.padding_x);
        r.margin = j.value("margin", def.margin);
        r.scale = j.value("scale", def.scale);
        r.center_window = j.value("center_window", def.center_window);
        r.clamp_window = j.value("clamp_window", def.clamp_window);
        r.rollback_window = j.value("rollback_window", def.rollback_window);
    }

    void from_json(const nlohmann::json &j, PoseFilter &f) {
        const auto def = xm::data::def::poseFilter();
        f.velocity = j.value("velocity", def.velocity);
        f.window = j.value("window", def.window);
        f.fps = j.value("fps", def.fps);
    }

    void from_json(const nlohmann::json &j, PoseUndistort &u) {
        const auto def = xm::data::def::poseUndistort();
        u.source = j.value("source", def.source);
        u.points = j.value("points", def.points);
        u.alpha = j.value("alpha", def.alpha);
    }

    void from_json(const nlohmann::json &j, PoseThresholds &t) {
        const auto def = xm::data::def::poseThresholds();
        t.detector = j.value("detector", def.detector);
        t.marks = j.value("marks", def.marks);
        t.pose = j.value("pose", def.pose);
        t.roi = j.value("roi", def.roi);
    }

    void from_json(const nlohmann::json &j, PoseModel &m) {
        const auto def = xm::data::def::poseModel();
        m.detector = j.value("detector", def.detector);
        m.body = j.value("body", def.body);
    }

    void from_json(const nlohmann::json &j, PoseDevice &d) {
        const auto def = xm::data::def::poseDevice();
        j.at("calibration").get_to(d.calibration);
        d.model = j.value("model", def.model);
        d.threshold = j.value("threshold", def.threshold);
        d.undistort = j.value("undistort", def.undistort);
        d.filter = j.value("filter", def.filter);
        d.roi = j.value("roi", def.roi);
    }

    void from_json(const nlohmann::json &j, Pose &p) {
        const auto def = xm::data::def::pose();
        j.at("devices").get_to(p.devices);
        j.at("stereo").get_to(p.stereo);
        p.show_epilines = j.value("show_epilines", def.show_epilines);
        p.segmentation = j.value("segmentation", def.segmentation);
        p.threads = j.value("threads", def.threads);
    }

    void from_json(const nlohmann::json &j, Misc &m) {
        m.cpu = j.value("cpu", 8);
        m.debug = j.value("debug", false);
    }

    void from_json(const nlohmann::json &j, JsonConfig &c) {
        j.at("type").get_to(c.type);
        j.at("camera").get_to(c.camera);
        c.misc = j.value("misc", xm::data::def::misc());
        c.gui = j.value("gui", xm::data::def::gui());
        c.pose = j.value("pose", xm::data::def::pose());

        if (c.type == ConfigType::CALIBRATION || c.type == ConfigType::CROSS_CALIBRATION) {
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

    std::filesystem::path create_dir_rec(const std::filesystem::path &path) {
        if (std::filesystem::exists(path))
            return path;
        if (!std::filesystem::create_directories(path))
            throw std::runtime_error("Cannot create directories: " + path.string());
        return path;
    }
}