#include <opencv2/opencv.hpp>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <iostream>
#include <algorithm> // For std::min, std::max
using namespace ::executorch::extension;

int main()
{
  // Load an image from disk using OpenCV.
  cv::Mat inputImage = cv::imread("sample.jpg");
  if (inputImage.empty()) {
    std::cerr << "Error: Could not load the input image!" << std::endl;
    return -1;
  }

  // Convert the image to float and normalize to [0, 1].
  inputImage.convertTo(inputImage, CV_32FC3, 1.0f / 255.0f);

  // Convert HWC (Height, Width, Channels) to CHW (Channels, Height, Width).
  std::vector<cv::Mat> chw(3);
  cv::split(inputImage, chw); // Split into separate channels (B, G, R)

  // Concatenate channels into a single tensor-friendly format.
  cv::Mat chwImage;
  cv::vconcat(chw, chwImage);

  // Create a tensor from the image data.
  auto tensor = from_blob(chwImage.data, {1, 3, inputImage.rows, inputImage.cols});

  // Create a Module and perform inference.
  Module module("xnnpack_simpleconv.pte");
  const auto result = module.forward(tensor);

  // Check for success or failure.
  if (!result.ok()) {
    std::cerr << "Error: Inference failed!" << std::endl;
    return -1;
  }

  // Retrieve the output tensor.
  const auto output = result->at(0).toTensor();

  // Get the shape of the output tensor.
  auto shape = output.sizes(); // Assuming shape is {1, C, H, W}
  int channels = shape[1];     // Number of channels
  int height = shape[2];       // Height
  int width = shape[3];        // Width

  // Print the shape of the output tensor.
  std::cout << "Output tensor shape: [";
  for (size_t i = 0; i < shape.size(); ++i) {
    std::cout << shape[i];
    if (i < shape.size() - 1) {
      std::cout << ", ";
    }
  }
  std::cout << "]" << std::endl;

  // Convert the output data to an OpenCV Mat.
  const float* outputData = output.const_data_ptr<float>();

  // Create an output image with multiple channels.
  std::vector<cv::Mat> outputChannels;
  for (int c = 0; c < channels; ++c) {
    cv::Mat channel(height, width, CV_32FC1, const_cast<float*>(outputData + c * height * width));
    outputChannels.push_back(channel.clone()); // Clone to ensure no memory issues
  }

  // Merge channels into a single image if needed (e.g., RGB).
  cv::Mat outputImage;
  cv::merge(outputChannels, outputImage);

  // Normalize the output image for display.
  outputImage.convertTo(outputImage, CV_8UC3, 255.0); // Convert to 8-bit for saving

  // Save the output image.
  cv::imwrite("output.jpg", outputImage);

  std::cout << "Inference complete. Output saved to 'output.jpg'." << std::endl;
  return 0;
}
