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

#include "../ocl/ocl_data.h"
#include "../utils/thread_pool.h"
#include "../../../platforms/agnostic_cap.h"

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
        bool rotate;
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

        /**
         * {id: cl_queue}
         */
        std::map<std::string, cl_command_queue> command_queues{};

        /**
         * {name: frame}
         */
        std::future<std::map<std::string, xm::ocl::Image2D>> buffer_future;

        /**
         * Camera properties
         */
        std::vector<SCamProp> properties{};

        /**
         * thread pool
         */
        std::shared_ptr<eox::util::ThreadPool> executor;

        bool fast = false;

    public:
        StereoCamera() = default;

        virtual ~StereoCamera();

        /**
        * This function releases any resources held by the current instance.
        */
        virtual void release();

        virtual void open(const SCamProp &prop);

        virtual std::map<std::string, xm::ocl::Image2D> captureWithName();

        virtual std::vector<xm::ocl::Image2D> capture();

        /**
         * Enqueue asynchronous frame capturing
         */
        virtual void enqueue();

        /**
         * @return asynchronously grabbed frames enqueued by calling "enqueue()"
         */
        virtual std::vector<xm::ocl::Image2D> dequeue();

        virtual std::map<std::string, xm::ocl::Image2D> dequeueWithName();

        virtual void setControl(const std::string &device_id, uint prop_id, int value);

        virtual void resetControls(const std::string &device_id);

        virtual void resetControls();

        virtual void setFastMode(bool fast);

        virtual void setThreadPool(std::shared_ptr<eox::util::ThreadPool> executor);

        [[nodiscard]] virtual bool getFastMode() const;

        [[nodiscard]] virtual std::vector<platform::cap::camera_controls> getControls() const;

        [[nodiscard]] virtual platform::cap::camera_controls getControls(const std::string &device_id) const;

        [[nodiscard]] uint getDeviceIndex(const std::string &device_id) const;

        virtual void save(std::ostream &output_stream, const std::string &device_id, const std::string &name) const;

        virtual void read(std::istream &input_stream, const std::string &device_id, const std::string &name);
    };

} // xm

#endif //XMOTION_STEREO_CAMERA_H
