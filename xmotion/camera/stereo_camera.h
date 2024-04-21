//
// Created by henryco on 4/18/24.
//

#ifndef XMOTION_STEREO_CAMERA_H
#define XMOTION_STEREO_CAMERA_H

#include <map>
#include <string>
#include <vector>

#include <spdlog/logger.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <opencv2/videoio.hpp>
#include "../utils/thread_pool.h"

namespace xm {

    namespace cap {

        typedef struct {
            uint id;
            std::string name;
            int min;
            int max;
            int step;
            int default_value;
            int value;
        } camera_control;

        typedef struct {
            /**
             * device id
             */
            std::string id;

            /**
             * device controls
             */
            std::vector<camera_control> controls;

        } camera_controls;

        int video_capture_api();

        int index_from_id(const std::string &id);

        camera_controls query_controls(const std::string &id);

        void set_control_value(const std::string &device_id, uint prop_id, int value);

        void save(std::ostream &output_stream, const std::string &name, const camera_controls &control);

        camera_controls read(std::istream &input_stream, const std::string &name);
    }

    typedef struct {
        std::string device_id;
        std::string codec;
        int width;
        int height;
        int fps;
        int buffer;
        bool flip_h;
        bool flip_v;
    } SCamProp;

    class StereoCamera {

        static inline const auto log =
                spdlog::stdout_color_mt("stereo_camera");

    protected:

        /**
         * {id: capture}
         */
        std::map<std::string, cv::VideoCapture> captures{};

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

        std::map<std::string, cv::Mat> captureWithId();

        std::vector<cv::Mat> capture();

        void setControl(const std::string &device_id, uint prop_id, int value);

        void resetControls(const std::string &device_id);

        void resetControls();

        void setFastMode(bool fast = true);

        void setThreadPool(std::shared_ptr<eox::util::ThreadPool> executor);

        [[nodiscard]] bool getFastMode() const;

        [[nodiscard]] std::vector<xm::cap::camera_controls> getControls() const;

        [[nodiscard]] xm::cap::camera_controls getControls(const std::string &device_id) const;

        [[nodiscard]] uint getDeviceIndex(const std::string &device_id) const;

        void save(std::ostream &output_stream, const std::string &device_id, const std::string &name) const;

        void read(std::istream &input_stream, const std::string &device_id, const std::string &name);
    };

} // xm

#endif //XMOTION_STEREO_CAMERA_H
