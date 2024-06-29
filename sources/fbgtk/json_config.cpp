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
            .capture_dummy = false,
            .capture_fast = false,
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
            .intrinsics = "",
            .threshold = xm::data::def::poseThresholds(),
            .undistort = xm::data::def::poseUndistort(),
            .filter = xm::data::def::poseFilter(),
            .model = xm::data::def::poseModel(),
            .roi = xm::data::def::poseRoi()
        };
    }

    ChainCalibration chainCalibration() {
        return {
            .files = {},
            .closed = false,
            ._present = false
        };
    }

    CrossCalibration crossCalibration() {
        return {
          ._present = false
        };
    }

    Pose pose() {
        return {
            .devices = {},
            .chain = xm::data::def::chainCalibration(),
            .cross = xm::data::def::crossCalibration(),
            .show_epilines = false,
            .segmentation = false,
            .threads = 0
        };
    }

    HSL hsl() {
        return {
          .h = 0,
          .s = 0,
          .l = 0
        };
    }

    BGR bgr() {
        return {
          .b = 0,
          .g = 0,
          .r = 0
        };
    }

    Blur blur() {
        return {
            .power = 1,
            ._present = false
        };
    }

    Difference difference() {
        return { ._present = false /*See original definition*/ };
    }

    Chroma chroma() {
        return {
            .key = "#00FF00",
            .replace = "#00FF00",
            .range = xm::data::def::hsl(),
            .blur = 0,
            .power = 0,
            .fine = 0,
            .refine = 0,
            .linear = false,
            ._present = false
        };
    }

    Filter filter() {
        return {
            .type = "",
            .blur = xm::data::def::blur(),
            .chroma = xm::data::def::chroma(),
            .difference = xm::data::def::difference()
        };
    }

    Compose compose() {
        return {
          .name = "",
          .chain = {}
        };
    }
}

namespace xm::data {
    NLOHMANN_JSON_SERIALIZE_ENUM(ConfigType, {
        { INVALID, nullptr },
        { CALIBRATION, "calibration" },
        { CROSS_CALIBRATION, "cross_calibration" },
        { CHAIN_CALIBRATION, "chain_calibration" },
        { COMPOSE, "compose" },
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

    void from_json(const nlohmann::json &j, HSL &h) {
        const auto def = xm::data::def::hsl();
        h.h = j.value("h", def.h);
        h.s = j.value("s", def.s);
        h.l = j.value("l", def.l);
    }

    void from_json(const nlohmann::json &j, Blur &b) {
        const auto def = xm::data::def::blur();
        b.power = j.value("power", def.power);
        b._present = true;
    }

    void from_json(const nlohmann::json &j, Difference &d) {
        const auto def = xm::data::def::difference();
        d.BASE_RESOLUTION = j.value("BASE_RESOLUTION", def.BASE_RESOLUTION);
        d.color = j.value("color", def.color);
        d.debug_on = j.value("debug_on", def.debug_on);
        d.adapt_on = j.value("adapt_on", def.adapt_on);
        d.ghost_on = j.value("ghost_on", def.ghost_on);
        d.lbsp_on = j.value("lbsp_on", def.lbsp_on);
        d.norm_l2 = j.value("norm_l2", def.norm_l2);
        d.linear = j.value("linear", def.linear);
        d.color_0 = j.value("color_0", def.color_0);
        d.lbsp_0 = j.value("lbsp_0", def.lbsp_0);
        d.lbsp_d = j.value("lbsp_d", def.lbsp_d);
        d.n_matches = j.value("n_matches", def.n_matches);
        d.t_upper = j.value("t_upper", def.t_upper);
        d.t_lower = j.value("t_lower", def.t_lower);
        d.model_size = j.value("model_size", def.model_size);
        d.ghost_l = j.value("ghost_l", def.ghost_l);
        d.ghost_n = j.value("ghost_n", def.ghost_n);
        d.ghost_n_inc = j.value("ghost_n_inc", def.ghost_n_inc);
        d.ghost_n_dec = j.value("ghost_n_dec", def.ghost_n_dec);
        d.alpha_d_min = j.value("alpha_d_min", def.alpha_d_min);
        d.alpha_norm = j.value("alpha_norm", def.alpha_norm);
        d.ghost_t = j.value("ghost_t", def.ghost_t);
        d.r_scale = j.value("r_scale", def.r_scale);
        d.r_cap = j.value("r_cap", def.r_cap);
        d.t_scale_inc = j.value("t_scale_inc", def.t_scale_inc);
        d.t_scale_dec = j.value("t_scale_dec", def.t_scale_dec);
        d.v_flicker_inc = j.value("v_flicker_inc", def.v_flicker_inc);
        d.v_flicker_dec = j.value("v_flicker_dec", def.v_flicker_dec);
        d.v_flicker_cap = j.value("v_flicker_cap", def.v_flicker_cap);
        d.refine_gate = j.value("refine_gate", def.refine_gate);
        d.refine_erode = j.value("refine_erode", def.refine_erode);
        d.refine_dilate = j.value("refine_dilate", def.refine_dilate);
        d.gate_threshold = j.value("gate_threshold", def.gate_threshold);
        d.kernel = j.value("kernel", def.kernel);
        d.gate_kernel = j.value("gate_kernel", def.gate_kernel);
        d.erode_kernel = j.value("erode_kernel", def.erode_kernel);
        d.dilate_kernel = j.value("dilate_kernel", def.dilate_kernel);
        d._present = true;
    }

    void from_json(const nlohmann::json &j, Chroma &c) {
        const auto def = xm::data::def::chroma();
        c.key = j.value("key", def.key);
        c.replace = j.value("replace", def.replace);
        c.range = j.value("range", def.range);
        c.blur = j.value("blur", def.blur);
        c.power = j.value("power", def.power);
        c.refine = j.value("fine", def.fine);
        c.refine = j.value("refine", def.refine);
        c.linear = j.value("linear", def.linear);
        c._present = true;
    }

    void from_json(const nlohmann::json &j, Filter &f) {
        const auto def = xm::data::def::filter();
        j.at("type").get_to(f.type);
        f.difference = def.difference;
        f.blur = def.blur;
        f.chroma = def.chroma;
        if (f.type == XM_FILTER_TYPE_BLUR)
            j.get_to(f.blur);
        else if (f.type == XM_FILTER_TYPE_CHROMA)
            j.get_to(f.chroma);
        else if (f.type == XM_FILTER_TYPE_DIFF)
            j.get_to(f.difference);
        else
            throw std::invalid_argument("Unknown filter type: " + f.type);
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

        c.filters = j.value("filters", std::vector<Filter>{});
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
        j.at("intrinsics").get_to(d.intrinsics);
        d.model = j.value("model", def.model);
        d.threshold = j.value("threshold", def.threshold);
        d.undistort = j.value("undistort", def.undistort);
        d.filter = j.value("filter", def.filter);
        d.roi = j.value("roi", def.roi);
    }

    void from_json(const nlohmann::json &j, ChainCalibration &c) {
        const auto def = xm::data::def::chainCalibration();
        j.at("files").get_to(c.files);
        c.closed = j.value("closed", def.closed);
        c._present = true;
    }

    void from_json(const nlohmann::json &j, CrossCalibration &c) {
        // TODO
        c._present = true;
    }

    void from_json(const nlohmann::json &j, Pose &p) {
        const auto def = xm::data::def::pose();
        j.at("devices").get_to(p.devices);
        p.chain = j.value("chain", def.chain);
        p.cross = j.value("cross", def.cross);
        p.show_epilines = j.value("epilines", def.show_epilines);
        p.segmentation = j.value("segmentation", def.segmentation);
        p.threads = j.value("threads", def.threads);
    }

    void from_json(const nlohmann::json &j, Misc &m) {
        const auto def = xm::data::def::misc();
        m.cpu = j.value("cpu", def.cpu);
        m.debug = j.value("debug", def.debug);
        m.capture_fast = j.value("capture_fast", def.capture_fast);
        m.capture_dummy = j.value("capture_dummy", def.capture_dummy);
    }

    void from_json(const nlohmann::json &j, Compose &c) {
        const auto def = xm::data::def::compose();
        c.name = j.value("name", def.name);
        c.chain = j.value("chain", def.chain);
    }

    void from_json(const nlohmann::json &j, JsonConfig &c) {
        j.at("type").get_to(c.type);
        j.at("captures").get_to(c.captures);

        c.misc = j.value("misc", xm::data::def::misc());
        c.gui = j.value("gui", xm::data::def::gui());
        c.pose = j.value("pose", xm::data::def::pose());
        c.compose = j.value("compose", xm::data::def::compose());

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

    std::vector<std::string> list_files(const std::filesystem::path &file) {
        if (!std::filesystem::exists(file))
            return {};
        if (std::filesystem::is_regular_file(file))
            return {file.string()};
        if (!std::filesystem::is_directory(file))
            return {};
        std::vector<std::string> files;
        for (const auto &entry: std::filesystem::directory_iterator(file)) {
            files.push_back(entry.path().string());
        }
        return files;
    }

    bool numeric_comparator_asc(const std::string &a, const std::string &b) {
        const auto extract_number = [](const std::string &f) -> int {
            const auto pos = f.find('.');
            if (pos != std::string::npos)
                return std::stoi(f.substr(0, pos));
            return std::stoi(f);
        };
        return extract_number(a) < extract_number(b);
    }


}