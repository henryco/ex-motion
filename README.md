# Work in progress...

### How to clone properly
```shell
git clone \
    --recursive-submodules \
    git@github.com:henryco/ex-motion.git
  ```

### Dependencies
- OpenGL (Should be present by default)
- OpenCL 3.0 (tl;dr: `opencl-headers`, `ocl-icd-libopencl1`, `nvidia-opencl-dev`)
- [OpenCV](https://opencv.org/get-started) (tl;dr: `libopencv-dev`)
- [GTKMM 4.0](https://gtkmm.org/en/download.html) (tl;dr: `libgtkmm-3.0-dev`, `libgtkmm-4.0-dev`)
- [Spdlog](https://github.com/gabime/spdlog) (tl;dr: `libspdlog-dev`)
- [V4l2](https://trac.gateworks.com/wiki/linux/v4l2) (tl;dr: `libv4l-dev`) (linux)
- [TensorFlow](https://github.com/tensorflow/tensorflow) (external/tensorflow)
- [Argparse](https://github.com/p-ranav/argparse#positional-arguments) (external/argparse)
- [Glm](https://github.com/g-truc/glm) (external/glm)

#### How to install dependencies (Debian/Nvidia)
```shell
apt install -y         \
    opencl-headers     \
    ocl-icd-libopencl1 \
    nvidia-opencl-dev  \
    libopencv-dev      \
    libgtkmm-3.0-dev   \
    libgtkmm-4.0-dev   \
    libspdlog-dev      \
    libv4l-dev
```