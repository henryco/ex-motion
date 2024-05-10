//
// Created by henryco on 5/10/24.
//

#ifndef XMOTION_D_DUMMY_CAMERA_H
#define XMOTION_D_DUMMY_CAMERA_H

#include "stereo_camera.h"
namespace xm {

    class DummyCamera : public xm::StereoCamera {

    protected:
        std::vector<cv::Mat> images;

    public:
        void release() override;

        ~DummyCamera() override = default;

        void open(const SCamProp &prop) override;

        std::map<std::string, cv::Mat> captureWithName() override;

        std::vector<cv::Mat> capture() override;

        void setControl(const std::string &device_id, uint prop_id, int value) override;

        void resetControls(const std::string &device_id) override;

        void resetControls() override;

        void setFastMode(bool fast) override;

        void setThreadPool(std::shared_ptr<eox::util::ThreadPool> executor) override;

        [[nodiscard]] bool getFastMode() const override;

        [[nodiscard]] std::vector<platform::cap::camera_controls> getControls() const override;

        [[nodiscard]] platform::cap::camera_controls getControls(const std::string &device_id) const override;

        void save(std::ostream &output_stream, const std::string &device_id, const std::string &name) const override;

        void read(std::istream &input_stream, const std::string &device_id, const std::string &name) override;
    };
}

#endif //XMOTION_D_DUMMY_CAMERA_H
