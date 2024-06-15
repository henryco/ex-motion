#include <filesystem>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>

std::string XM_removeDirectoriesAndExtension(const std::string& filename) {
    size_t lastSlash = filename.find_last_of("/\\");
    size_t pos = filename.rfind(".h");
    if (pos != std::string::npos) {
        if (lastSlash != std::string::npos) {
            return filename.substr(lastSlash + 1, pos - lastSlash - 1);
        } else {
            return filename.substr(0, pos);
        }
    }
    return filename;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input_file> <output_file>" << std::endl;
        return 1;
    }

    const auto input_filename = std::string(argv[1]);
    const auto output_filename = std::string(argv[2]);
    const auto array_name = XM_removeDirectoriesAndExtension(input_filename);

    std::filesystem::path f_path = output_filename;
    std::filesystem::path dir_path = f_path.parent_path();

    if (!std::filesystem::exists(dir_path)) {
        if (std::filesystem::create_directories(dir_path))
            std::cout << "Created directories: " << dir_path << '\n';
        else
            std::cout << "Directories already exist or failed to create" << '\n';
    }

    std::ifstream input_file(input_filename, std::ios::binary);
    if (!input_file.is_open()) {
        std::cerr << "Failed to open input file: " << input_filename << std::endl;
        return 1;
    }

    char c;
    std::ostringstream oss;
    while (input_file.get(c)) {
        oss << "0x" << std::hex << std::setw(2) << std::setfill('0') << (static_cast<unsigned>(c) & 0xFF) << ",";
    }

    std::ofstream output_file(output_filename, std::ios::out | std::ios::trunc);
    if (!output_file.is_open()) {
        std::cerr << "Failed to open output file: " << output_filename << std::endl;
        return 1;
    }

    std::string array_content = oss.str();
    if (!array_content.empty()) {
        array_content.pop_back();  // Remove the last comma
    }

    output_file << "#include \"" << array_name << ".h\"\n\n";
    output_file << "const char kernel_" << array_name << "_code[] = {" << array_content << "};\n";
    output_file << "const size_t kernel_" << array_name << "_code_size = sizeof(kernel_" << array_name << "_code);\n";

    return 0;
}