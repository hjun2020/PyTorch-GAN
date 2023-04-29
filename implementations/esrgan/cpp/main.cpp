#include <iostream>
#include <vector>
#include "onnxruntime_cxx_api.h"

// int main() {


//     Ort::AllocatorWithDefaultOptions allocator;

//     // Define the shape of the tensor
//     std::vector<int64_t> shape = {2, 3};

//     // Create a buffer for the tensor data
//     std::vector<float> tensor_data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

// Ort::Value tensor = Ort::Value::CreateTensor<float>(
//     Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault),
//     tensor_data.data(),
//     tensor_data.size(),
//     shape.data(),
//     shape.size()
// );


//     // Get the tensor shape and type
//     // auto tensor_shape = tensor.GetTensorTypeAndShapeInfo().GetShape();
//     // auto tensor_type = tensor.GetTensorTypeAndShapeInfo().GetElementType();
    
// // Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "my_env");
// // Ort::SessionOptions session_options;
// // session_options.SetIntraOpNumThreads(1);
// // session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);

// // Ort::Session session(env, tensor_data.data(), tensor_data.size(), session_options);

// }


int main() {
    // Initialize the ONNX Runtime environment
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");

    // Create a session options object
    Ort::SessionOptions session_options;

    // Create a session object
    const char* model_path = "../generator_ex.onnx";
    Ort::Session session(env, model_path, session_options);

    // Get the input and output names
    std::vector<const char*> input_names = {"input"};
    std::vector<const char*> output_names = {"output"};

    // Create input and output tensors
    std::vector<float> input_data(3 * 64 * 64, 1.0);
    std::vector<int64_t> input_shape = {1, 3, 64, 64};
    std::vector<float> output_data(3 * 256 * 256, 1.0);
    std::vector<int64_t> output_shape = {1, 3, 256, 256};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault), input_data.data(), input_data.size(), input_shape.data(), input_shape.size());
    Ort::Value output_tensor = Ort::Value::CreateTensor<float>(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault), output_data.data(), output_data.size(), output_shape.data(), output_shape.size());

    // session.Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1, output_names.data(), 1, &output_tensor);
    session.Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, input_names.size(), output_names.data(), &output_tensor, output_names.size());

    float* output_data1 = output_tensor.GetTensorMutableData<float>();
    for (int i = 0; i < 10; i++) {
        std::cout << output_data1[i] << std::endl;
    }

    return 0;
}
    // Run the model

    // const char* input_name = "input";
    // const char* output_name = "output";
    // std::vector<const char*> input_names = { input_name };
    // std::vector<const char*> output_names = { output_name };
    // std::vector<Ort::Value> input_tensors = { input_tensor };
    // std::vector<Ort::Value> output_tensors = { output_tensor };
    // session.Run(Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(), input_names.size(), output_names.data(), output_tensors.size(), output_tensors.data());

    // session.Run(Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(), input_names.size(), output_names.data(), output_tensors.size(), output_tensors.data());
    // session.Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, input_names.size(), output_names.data(), output_tensor.size(), &output_tensor);

//     // // Print the output
//     // float* output_data = output_tensor.GetTensorMutableData<float>();
//     // for (int i = 0; i < 10; i++) {
//     //     std::cout << output_data[i] << std::endl;
//     // }

//     return 0;


