//
// Created by henryco on 4/25/24.
//

#include <filesystem>
#include <iomanip>
#include <opencv2/core/persistence.hpp>
#include <ctime>
#include <chrono>
#include "../../xmotion/fbgtk/data/json_ocv.h"

void xm::data::ocv::write_calibration(const std::string &file, const xm::data::ocv::Calibration &c) {
    cv::FileStorage fs(file, cv::FileStorage::WRITE);
    {
        fs << "type" << "single_calibration";
        fs << "name" << c.name;
        fs << "timestamp" << utc_iso_date_str_now();
        fs << "K" << c.K;
        fs << "D" << c.D;
        fs << "width" << c.width;
        fs << "height" << c.height;
        fs << "fov_x" << c.fov_x;
        fs << "fov_y" << c.fov_y;
        fs << "f" << c.f;
        fs << "center_x" << c.c_x;
        fs << "center_y" << c.c_y;
        fs << "aspect_ratio" << c.r;
        fs << "error" << c.error;
    }
    fs.release();
}

void xm::data::ocv::write_chain_calibration(const std::string &file, const xm::data::ocv::ChainCalibration &c) {
    cv::FileStorage fs(file, cv::FileStorage::WRITE);
    {
        fs << "type" << "chain_calibration";
        fs << "name" << c.name;
        fs << "timestamp" << utc_iso_date_str_now();
        fs << "R" << c.R;
        fs << "T" << c.T;
        fs << "E" << c.E;
        fs << "F" << c.F;
        fs << "RT" << c.RT;
        fs << "error" << c.error;
    }
    fs.release();
}

xm::data::ocv::Calibration xm::data::ocv::read_calibration(const std::string &file) {
    if (!std::filesystem::exists(file))
        throw std::runtime_error("File: " + file + " does not exists!");

    Calibration c;
    cv::FileStorage fs(file, cv::FileStorage::READ);
    {
        fs["type"] >> c.type;
        fs["name"] >> c.name;
        fs["timestamp"] >> c.timestamp;
        fs["K"] >> c.K;
        fs["D"] >> c.D;
        fs["width"] >> c.width;
        fs["height"] >> c.height;
        fs["fov_x"] >> c.fov_x;
        fs["fov_y"] >> c.fov_y;
        fs["f"] >> c.f;
        fs["center_x"] >> c.c_x;
        fs["center_y"] >> c.c_y;
        fs["aspect_ratio"] >> c.r;
        fs["error"] >> c.error;
    }
    fs.release();
    return c;
}

xm::data::ocv::ChainCalibration xm::data::ocv::read_chain_calibration(const std::string &file) {
    if (!std::filesystem::exists(file))
        throw std::runtime_error("File: " + file + " does not exists!");

    ChainCalibration c;
    cv::FileStorage fs(file, cv::FileStorage::READ);
    {
        fs["name"] >> c.name;
        fs["R"] >> c.R;
        fs["T"] >> c.T;
        fs["E"] >> c.E;
        fs["F"] >> c.F;
        fs["RT"] >> c.RT;
        fs["error"] >> c.error;
    }
    return c;
}

std::string xm::data::ocv::utc_iso_date_str_now() {
    const auto now = std::chrono::system_clock::now();
    const auto time_t_now = std::chrono::system_clock::to_time_t(now);
    const auto tm_now = std::gmtime(&time_t_now);
    std::ostringstream os;
    os << std::put_time(tm_now, "%FT%TZ");
    return os.str();
}
