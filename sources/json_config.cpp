//
// Created by henryco on 4/18/24.
//

#include "../xmotion/data/json_config.h"
#include <fstream>


xm::data::JsonConfig xm::data::config_from_file(const std::string &path) {
    nlohmann::json content;

    std::ifstream file(path);
    if (!file.is_open())
        throw std::runtime_error("Cannot open file: " + path);

    file >> content;
    file.close();

    return content.template get<xm::data::JsonConfig>();
}

std::string xm::data::prepare_project_file(const std::string &path) {
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

std::string xm::data::prepare_project_file(const char *c_str) {
    return prepare_project_file(std::string(c_str));
}
