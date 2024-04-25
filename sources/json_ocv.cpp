//
// Created by henryco on 4/25/24.
//

#include <opencv2/core/persistence.hpp>
#include "../xmotion/data/json_ocv.h"

void xm::data::ocv::write_calibration(const std::string &file, const xm::data::ocv::Calibration &c) {
    cv::FileStorage fs(file, cv::FileStorage::WRITE);
    {
        fs << "name" << c.name;
        fs << "K" << c.K;
        fs << "D" << c.D;
        fs << "error" << c.error;
    }
    fs.release();
}

xm::data::ocv::Calibration xm::data::ocv::read_calibration(const std::string &file) {
    Calibration c;
    cv::FileStorage fs(file, cv::FileStorage::READ);
    {
        fs["name"] >> c.name;
        fs["K"] >> c.K;
        fs["D"] >> c.D;
        fs["error"] >> c.error;
    }
    fs.release();
    return c;
}
