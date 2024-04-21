//
// Created by henryco on 4/18/24.
//

#ifndef XMOTION_STEREO_CAMERA_H
#define XMOTION_STEREO_CAMERA_H

#include <map>
#include <string>
#include <vector>

#include <fstream>
#include <spdlog/logger.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <opencv2/videoio.hpp>

#include "../utils/thread_pool.h"
#include "../../platforms/agnostic_cap.h"

namespace xm {

    typedef struct {
        std::string device_id;
        std::string name;
        std::string codec;
        int width;
        int height;
        int fps;
        int buffer;
        bool flip_x;
        bool flip_y;
        int x;
        int y;
        int w;
        int h;
    } SCamProp;

    class StereoCamera {

        static inline const auto log =
                spdlog::stdout_color_mt("stereo_camera");

    protected:

        /**
         * {id: capture}
         */
        std::map<std::string, cv::VideoCapture> captures{};

        std::vector<SCamProp> properties{};

        /**
         * threadpool
         */
        std::shared_ptr<eox::util::ThreadPool> executor;

        bool fast = false;

    public:
        StereoCamera() = default;

        ~StereoCamera();

        /**
        * This function releases any resources held by the current instance.
        */

        void release();

        void open(const SCamProp &prop);

        std::map<std::string, cv::Mat> captureWithName();

        std::vector<cv::Mat> capture();

        void setControl(const std::string &device_id, uint prop_id, int value);

        void resetControls(const std::string &device_id);

        void resetControls();

        void setFastMode(bool fast = true);

        void setThreadPool(std::shared_ptr<eox::util::ThreadPool> executor);

        [[nodiscard]] bool getFastMode() const;

        [[nodiscard]] std::vector<platform::cap::camera_controls> getControls() const;

        [[nodiscard]] platform::cap::camera_controls getControls(const std::string &device_id) const;

        [[nodiscard]] uint getDeviceIndex(const std::string &device_id) const;

        void save(std::ostream &output_stream, const std::string &device_id, const std::string &name) const;

        void read(std::istream &input_stream, const std::string &device_id, const std::string &name);
    };

} // xm

#endif //XMOTION_STEREO_CAMERA_H
