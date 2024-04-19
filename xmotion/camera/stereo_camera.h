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

    typedef struct {
        uint id;
        uint type;
        std::string name;
        int min;
        int max;
        int step;
        int default_value;
        int value;
    } camera_control;

    typedef struct {
        std::string id;
        std::vector<camera_control> controls;
    } camera_controls;

    int video_capture_api();

    class StereoCamera {

        static inline const auto log =
                spdlog::stdout_color_mt("stereo_camera");

    protected:

        std::map<std::string, cv::VideoCapture> captures;
        std::shared_ptr<eox::util::ThreadPool> executor;

        bool fast = false;

    public:
        StereoCamera() = default;

        ~StereoCamera();

        /**
        * This function releases any resources held by the current instance.
        */

        void release();

        /**
         * @see eox::StereoCamera::open(std::vector<CameraProp> props)
         */
        void open();
    };

} // xm

#endif //XMOTION_STEREO_CAMERA_H
