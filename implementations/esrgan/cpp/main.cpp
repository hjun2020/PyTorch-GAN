#include <iostream>
#include <vector>
#include "onnxruntime_cxx_api.h"

int main() {
  // Create session options and load the model



    Ort::AllocatorWithDefaultOptions allocator;

    // Define the shape of the tensor
    std::vector<int64_t> shape = {2, 3};

    // Create a buffer for the tensor data
    std::vector<float> tensor_data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

Ort::Value tensor = Ort::Value::CreateTensor<float>(
    Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault),
    tensor_data.data(),
    tensor_data.size(),
    shape.data(),
    shape.size()
);


    // Get the tensor shape and type
    auto tensor_shape = tensor.GetTensorTypeAndShapeInfo().GetShape();
    auto tensor_type = tensor.GetTensorTypeAndShapeInfo().GetElementType();
    
Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "my_env");
Ort::SessionOptions session_options;
session_options.SetIntraOpNumThreads(1);
session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);

Ort::Session session(env, tensor_data.data(), tensor_data.size(), session_options);



  return 0;
}
