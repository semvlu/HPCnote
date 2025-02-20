#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

#define CHECK_ERR(x) if (x != CL_SUCCESS) { std::cerr << "OpenCL Error: " << x << std::endl; exit(1); }

const char* kernelSource = R"(
__kernel void computeZernike(__global float *image, __global float *momentOutput, 
                             int width, int height, int order) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height) return;

    float Z_real = 0.0f, Z_imag = 0.0f;
    float rho, theta;
    
    int idx = y * width + x;
    float pixel = image[idx];

    float cx = width / 2.0f, cy = height / 2.0f;
    float norm_x = (x - cx) / cx;
    float norm_y = (y - cy) / cy;
    rho = sqrt(norm_x * norm_x + norm_y * norm_y);
    theta = atan2(norm_y, norm_x);

    if (rho > 1.0f) return;

    float radial = (2 * pow(rho, 4) - 3 * rho * rho + 1);
    float basis_real = radial * cos(2 * theta);
    float basis_imag = radial * sin(2 * theta);

    Z_real = pixel * basis_real;
    Z_imag = pixel * basis_imag;

    momentOutput[idx] = sqrt(Z_real * Z_real + Z_imag * Z_imag);
}
)";

int main() {
    std::string imgPath = "C:/Users/David/Desktop/zernikeEdgeDetect/assets/genesis.jpg";
    cv::Mat image = cv::imread(imgPath, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error: Could not load image." << std::endl;
        return -1;
    }
    int width = image.cols;
    int height = image.rows;
    std::vector<float> imageData(width * height);
    std::vector<float> result(width * height, 0);

    // Convert image to float
    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++)
            imageData[i * width + j] = image.at<uchar>(i, j) / 255.0f;

    // OpenCL setup
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_mem imageBuffer, resultBuffer;

    CHECK_ERR(clGetPlatformIDs(1, &platform, NULL));
    CHECK_ERR(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL));

    context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    queue = clCreateCommandQueueWithProperties(context, device, 0, NULL);

    program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, NULL);
    CHECK_ERR(clBuildProgram(program, 0, NULL, NULL, NULL, NULL));

    kernel = clCreateKernel(program, "computeZernike", NULL);

    imageBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        width * height * sizeof(float), imageData.data(), NULL);
    resultBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        width * height * sizeof(float), NULL, NULL);

    CHECK_ERR(clSetKernelArg(kernel, 0, sizeof(cl_mem), &imageBuffer));
    CHECK_ERR(clSetKernelArg(kernel, 1, sizeof(cl_mem), &resultBuffer));
    CHECK_ERR(clSetKernelArg(kernel, 2, sizeof(int), &width));
    CHECK_ERR(clSetKernelArg(kernel, 3, sizeof(int), &height));
    int order = 4;
    CHECK_ERR(clSetKernelArg(kernel, 4, sizeof(int), &order));

    size_t globalSize[] = { static_cast<size_t>(width), static_cast<size_t>(height) };
    CHECK_ERR(clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, NULL, 0, NULL, NULL));

    CHECK_ERR(clEnqueueReadBuffer(queue, resultBuffer, CL_TRUE, 0, width * height * sizeof(float),
        result.data(), 0, NULL, NULL));

    // Convert result to an image
    cv::Mat edgeImage(height, width, CV_8UC1);
    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++)
            edgeImage.at<uchar>(i, j) = static_cast<uchar>(result[i * width + j] * 255);

    cv::imwrite("output.jpg", edgeImage);

    // Cleanup
    clReleaseMemObject(imageBuffer);
    clReleaseMemObject(resultBuffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    std::cout << "Edge detection complete. Output saved as output.jpg" << std::endl;
    return 0;
}
