#include "argparse/argparse.hpp"
#include <spdlog/spdlog.h>
#include <filesystem>
#include <execinfo.h>

#include "xmotion/boot/boot.h"
#include "xmotion/boot/gtk_boot.h"
#include "xmotion/boot/file_boot.h"

namespace xm::error {

#ifdef _WIN32
    void printStackTrace() {
        // Honestly I have no idea if this would work

        HANDLE process = GetCurrentProcess();
        SymInitialize(process, NULL, TRUE);

        void* stack[100];
        unsigned short frames = CaptureStackBackTrace(0, 100, stack, NULL);

        SYMBOL_INFO* symbol = (SYMBOL_INFO*)calloc(sizeof(SYMBOL_INFO) + 256 * sizeof(char), 1);
        symbol->MaxNameLen = 255;
        symbol->SizeOfStruct = sizeof(SYMBOL_INFO);

        for (unsigned int i = 0; i < frames; i++) {
            SymFromAddr(process, (DWORD64)(stack[i]), 0, symbol);
            std::cerr << frames - i - 1 << ": " << symbol->Name << " at 0x" << std::hex << symbol->Address << std::dec << std::endl;
        }

        free(symbol);
    }
#else
    void printStackTrace() {
        void* array[10];

        // get void*'s for all entries on the stack
        const int size = backtrace(array, 10);

        // print out all the frames
        char** messages = backtrace_symbols(array, size);

        for (size_t i = 0; i < size; ++i) { // Using size_t for loop counter
            std::cerr << messages[i] << std::endl;
        }

        free(messages);
    }
}
#endif

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
    } catch (std::exception &e) {
        std::cerr << "Unexpected Exception: " << e.what() << '\n';
        xm::error::printStackTrace();
        return 1;
    } catch (...) {
        std::cerr << "Debug Report: An exception occurred" << '\n';
        xm::error::printStackTrace();
        return 1;
    }

    return 0;
}
