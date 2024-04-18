#include "argparse/argparse.hpp"
#include <spdlog/spdlog.h>
#include <filesystem>

#include "xmotion/boot/boot.h"
#include "xmotion/boot/gtk_boot.h"
#include "xmotion/boot/file_boot.h"

int main(int argc, char **argv) {

    argparse::ArgumentParser program("xmotion", "0.0.1");
    program.add_description(R"desc(
                To list available capture devices on linux you can use:
                [  $ v4l2-ctl --list-devices  ]

                For proper configuration first check your camera allowed properties
                [  $ v4l2-ctl -d \"/dev/video${ID}\" --list-formats-ext  ]
                )desc");
    program.add_argument("-p", "--project")
            .default_value(std::string(std::filesystem::current_path().string()))
            .help("Project root directory");
    program.add_argument("-g", "--graphic")
            .help("Graphic mode")
            .flag();
    program.add_argument("-v", "--verbose")
            .help("Increase output verbosity")
            .flag();


    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error &err) {
        std::cout << err.what() << '\n';
        std::cout << program;
        return 1;
    }

    if (program.get<bool>("--verbose")) {
        spdlog::set_level(spdlog::level::debug);
    } else {
        spdlog::set_level(spdlog::level::info);
    }

    std::unique_ptr<xm::Boot> director;

    if (program.get<bool>("--graphic")) {
        director = std::make_unique<xm::GtkBoot>();
    } else {
        director = std::make_unique<xm::FileBoot>();
    }

    try {
        director->open_project(program.get<std::string>("--project").c_str());
        return director->boot(argc, argv);
    } catch (...) {
        std::cerr << "Debug Report: An exception occurred" << '\n';
    }

    return 0;
}
