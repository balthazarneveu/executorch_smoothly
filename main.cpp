#include <opencv2/opencv.hpp>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <iostream>
#include <vector>
using namespace ::executorch::extension;

int main()
{
    // Open webcam
    cv::VideoCapture cap(0); // 0 is usually the default webcam
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open webcam!" << std::endl;
        return -1;
    }

    // Load the model
    Module module("xnnpack_simpleconv.pte");

    // Create a window to display the results
    cv::namedWindow("Webcam Output", cv::WINDOW_AUTOSIZE);

    bool isProcessing = true; // Toggle flag for processing

    while (true) {
        cv::Mat frame;
        cap >> frame; // Capture a frame from the webcam

        if (frame.empty()) {
            std::cerr << "Error: Could not capture frame!" << std::endl;
            break;
        }

        if (isProcessing) {
            // Preprocess the frame (convert to float and normalize)
            frame.convertTo(frame, CV_32FC3, 1.0f / 255.0f);

            // Convert HWC to CHW
            std::vector<cv::Mat> chw(3);
            cv::split(frame, chw);
            cv::Mat chwImage;
            cv::vconcat(chw, chwImage);

            // Create a tensor from the image data
            auto tensor = from_blob(chwImage.data, {1, 3, frame.rows, frame.cols});

            // Perform inference
            const auto result = module.forward(tensor);

            if (!result.ok()) {
                std::cerr << "Error: Inference failed!" << std::endl;
                break;
            }

            // Retrieve the output tensor
            const auto output = result->at(0).toTensor();

            // Get the output shape
            auto shape = output.sizes(); // Assuming shape is {1, C, H, W}
            int channels = shape[1];
            int height = shape[2];
            int width = shape[3];

            // Convert the output tensor to OpenCV Mat
            const float* outputData = output.const_data_ptr<float>();
            std::vector<cv::Mat> outputChannels;
            for (int c = 0; c < channels; ++c) {
                cv::Mat channel(height, width, CV_32FC1, const_cast<float*>(outputData + c * height * width));
                outputChannels.push_back(channel.clone());
            }

            cv::Mat outputImage;
            cv::merge(outputChannels, outputImage);

            // Normalize and convert to 8-bit for display
            outputImage.convertTo(outputImage, CV_8UC3, 255.0);

            // Display the processed output
            cv::imshow("Webcam Output", outputImage);
        } else {
            // Simply show the raw frame if processing is off
            cv::imshow("Webcam Output", frame);
        }

        // Keyboard control for toggling and exiting
        char key = (char)cv::waitKey(1);
        if (key == 'q') {
            // Exit the loop
            break;
        } else if (key == 't') {
            // Toggle processing state
            isProcessing = !isProcessing;
            std::cout << "Processing " << (isProcessing ? "enabled" : "disabled") << "." << std::endl;
        }
    }

    // Cleanup
    cap.release();
    cv::destroyAllWindows();

    return 0;
}
