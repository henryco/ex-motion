//
// Created by henryco on 12/20/23.
//

#include "../../xmotion/core/ocl/kernel.h"

namespace eox::ocl {

    void Kernel::compile(const std::string& _kernel, const std::string& flags) {
//        auto context = cv::ocl::Context::getDefault(true);

        std::string err_msg;
        source = std::make_unique<cv::ocl::ProgramSource>(_kernel);
        program = std::make_unique<cv::ocl::Program>(*source, flags, err_msg);
        if (!err_msg.empty()) {
            log->error("kernel error: {}", err_msg);
            throw std::runtime_error(err_msg);
        }
    }

    cv::ocl::Kernel &Kernel::get_kernel(const std::string &name) {
        if (kernels.contains(name))
            return kernels[name];

        kernels.emplace(name, cv::ocl::Kernel(name.c_str(), *program));
        return kernels[name];
    }

    cv::ocl::Kernel &Kernel::get_kernel() {
        return kernels.begin()->second;
    }

    cv::ocl::Kernel& Kernel::procedure(const std::string &name, const std::string &opts) {
        std::string err_msg;
        kernels.emplace(name, cv::ocl::Kernel(name.c_str(), *source, opts, &err_msg));
        if (!err_msg.empty()) {
            log->error("kernel error: {}", err_msg);
            throw std::runtime_error(err_msg);
        }

        size_t pref = kernels[name].preferedWorkGroupSizeMultiple();
        pref_size.emplace(name, pref);

        return kernels[name];
    }

    size_t Kernel::get_pref_size(const std::string &name) {
        if (pref_size.contains(name))
            return pref_size[name];

        pref_size.emplace(name, cv::ocl::Kernel(name.c_str(), *program).preferedWorkGroupSizeMultiple());
        return pref_size[name];
    }

    size_t Kernel::get_pref_size() {
        return pref_size.begin()->second;
    }

    cv::ocl::Kernel &Kernel::get_kernel(int idx) {
        int i = 0;
        for (auto &pair: kernels) {
            if (i == idx)
                return pair.second;
            i++;
        }
        throw std::out_of_range("IDX >= kernels.size()");
    }


} // eox