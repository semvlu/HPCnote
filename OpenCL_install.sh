# OpenCL install

git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
bootstrap-vcpkg.bat

vcpkg install <lib> # in casu opencl
vcpkg integrate install # integrate w/ VS 2022