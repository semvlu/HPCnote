#include <CL/cl2.hpp>
#include <iostream>
#include <vector>

#define SIZE 10

/*
    OpenCL tutorial by Universite du Luxembourg (UL) HPC
    https://ulhpc-tutorials.readthedocs.io/en/latest/gpu/opencl/
    https://github.com/ULHPC/tutorials/tree/devel/gpu/opencl/code

    OpenCL Error code reference: https://gist.github.com/bmount/4a7144ce801e5569a0b6
*/

// Arch Hierarchy: Host >= Global Mem > Device > Compute Unit (Work-group) > Processing Element (Work-item) 
// Work-group allows synchronisation: barrier & mem. fence

// Kernel: string for hi portability, appended to prog @runtime
// const char* / std::string
// R"()": syntax for raw string
// kern func: void. global: global mem
// kern arg: global, local, constant
const char* kern = R"(
 __kernel void kern0(global const int* A, global const int* B, global int* C) {
    int i = get_global_id(0);
    C[i] = A[i] + B[i];
}
)";


int main() {
    // ------ Sys env ------
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);

    cl::Platform plat = all_platforms[0];
    // Platform
    std::cout << plat.getInfo<CL_PLATFORM_NAME>() << std::endl;


    std::vector<cl::Device> all_devices;
    plat.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    if (all_devices.size() == 0) {
        std::cout << " No devices found.\n";
        exit(1);
    }

    for (const auto& device : all_devices) {
        printDeviceInfo(device);
    }

    cl::Device dev = all_devices[0];
    std::cout << "Using device: " << dev.getInfo<CL_DEVICE_NAME>() << std::endl;
    std::cout << "#CU: " << dev.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
    std::cout << "Memory size: " << dev.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() << std::endl;


    // ------ Create context & sources ------
    
    // Context: inter-device mem. share
    cl::Context contxt({ dev });
    cl::Program::Sources src;

    // ------ Input ------
    // Host 
    int A_h[SIZE] = { 0,1,2,3,4,5,6,7,8,9 };
    int B_h[SIZE] = { 10,9,8,7,6,5,4,3,2,1 };

    // ------ Buffer setup ------

    // Buffer: mem allo. to the dev.
    // Image: 2D/3D buffer
    cl::Buffer buf_A(contxt, CL_MEM_READ_ONLY, sizeof(int) * SIZE);
    cl::Buffer buf_B(contxt, CL_MEM_READ_ONLY, sizeof(int) * SIZE);
    
    cl::Buffer buf_C(contxt, CL_MEM_READ_WRITE, sizeof(int) * SIZE);
    // CL_MEM_READ(WRITE)_ONLY / CL_MEM_READ_WRITE


    // ------ Command (Task) Queue ------
    // Queue: push cmd onto Dev, ~= CUDA streams
    // queue for each dev
    cl::CommandQueue qu(contxt, dev);
    // Read/Write/Map/Copy
    // blocking = CL_TRUE for sync
    qu.enqueueWriteBuffer(buf_A, CL_TRUE, 0, sizeof(int) * SIZE, A_h);
    qu.enqueueWriteBuffer(buf_B, CL_TRUE, 0, sizeof(int) * SIZE, B_h);
	
    // ------ Run kernel in source ------

    // push kern to the source code
    src.push_back({ kern, strlen(kern) });
    // OpenCL prog: link src to context
    cl::Program prog(contxt, src);

    // check build
    if (prog.build({ dev }) != CL_SUCCESS) {
        std::cout << " Error building: " << prog.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev) << std::endl;
        exit(1);
    }
    // get_global_id, get_global_size, get_work_dim, get+
    // https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/get_work_dim.html
    // https://community.khronos.org/t/when-to-use-get-global-id-and-get-local-id-in-opencl/3999/4


    // ------ Create kernel for exec ------
    cl::compatibility::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Event> kern0(cl::Kernel(prog, "kern0"));
    // *N.B.* Kernel name must match the function name
    // https://github.khronos.org/OpenCL-CLHPP/structcl_1_1compatibility_1_1make__kernel.html

    cl::NDRange global(SIZE); // #threads on dev

    // Event
    cl::Event ekern0;


    kern0( cl::EnqueueArgs(qu, global),
        buf_A, buf_B, buf_C, ekern0).wait();
    // .wait() or .waitForEvents( {ekern0} )

    cl_ulong start = ekern0.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    cl_ulong end = ekern0.getProfilingInfo<CL_PROFILING_COMMAND_END>();

    std::cout << "kern0 Execution Time (ns): " << (end - start) << std::endl;
    
    
    
    int C_h[SIZE];

    // Read data from dev: from buf_C -> C_h
    qu.enqueueReadBuffer(buf_C, CL_TRUE, 0, sizeof(int) * SIZE, C_h);

    std::cout << "---------- Host C ----------" << std::endl;
    for (int i = 0; i < SIZE; i++) {
        std::cout << C_h[i] << std::endl;
    }
    std::cout << "----------------------------" << std::endl;


    // ------ Events: inter-queue sync ------

    return 0;
}


void printDeviceInfo(const cl::Device& device) {
    std::cout << "  Device Name: " << device.getInfo<CL_DEVICE_NAME>() << "\n";
    std::cout << "  Device Type: ";
    switch (device.getInfo<CL_DEVICE_TYPE>()) {
    case CL_DEVICE_TYPE_CPU: std::cout << "CPU"; break;
    case CL_DEVICE_TYPE_GPU: std::cout << "GPU"; break;
    case CL_DEVICE_TYPE_ACCELERATOR: std::cout << "Accelerator"; break;
    default: std::cout << "Other"; break;
    }
    std::cout << "\n";

    std::cout << "  Compute Units: " << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << "\n";
    std::cout << "  Global Memory: " << device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() / (1024 * 1024) << " MB\n";
    std::cout << "  Local Memory: " << device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() / 1024 << " KB\n";
    std::cout << "  Max Work Group Size: " << device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << "\n";
    std::cout << "  Max Clock Frequency: " << device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() << " MHz\n";
    std::cout << "----------------------------------------\n";
}

