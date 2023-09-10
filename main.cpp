#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>

#include <eigen3/Eigen/Eigen>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>

#include <random>
std::default_random_engine generator;
std::uniform_real_distribution<double> distribution(0.0, 1.0);

#include "include/project_path.h"

using std::cout;
using std::endl;

// Data set image size: 28x28 pixels, total 784
// sigmoid 1/(1+e^(-x))

const cv::Size size_raw(28, 28);   // original one number
const cv::Size size_big(250, 250); // upscaled one number
// const cv::Size size_raw(28, 28 * 10);   // original row of numbers
// const cv::Size size_big(100, 100 * 10); // upscaled row of numbers
// const cv::Size size_raw(28 * 3, 28 * 1);   // original matrix of numbers
// const cv::Size size_big(100 * 3, 100 * 1); // upscaled matrix of numbers

Eigen::Matrix<float, 28 * 28, 1> a1;
Eigen::Matrix<float, 10, 1> a2; // hidden layer
Eigen::Matrix<float, 10, 1> a3; // oputput layer
Eigen::Matrix<float, 10, 1> z2;
Eigen::Matrix<float, 10, 1> z3;
Eigen::Matrix<float, 10, 28 * 28> w2;
Eigen::Matrix<float, 10, 10> w3;
Eigen::Matrix<float, 10, 1> b2;
Eigen::Matrix<float, 10, 1> b3;

void getImage(char* file_array, size_t image_number, cv::Mat& image)
{
  memcpy(image.data, file_array + 4 * sizeof(uint32_t) + image_number * size_raw.area(), size_raw.width * size_raw.height);
}

void getLabel(char* file_array, size_t image_number, uint8_t& label)
{
  memcpy(&label, file_array + 2 * sizeof(uint32_t) + image_number, 1);
}

void getInput(char* file_array, size_t image_number, Eigen::Matrix<float, 28 * 28, 1>& input)
{
  static Eigen::Matrix<uint8_t, 28 * 28, 1> tmp;

  memcpy(tmp.data(), file_array + 4 * sizeof(uint32_t) + image_number * size_raw.area(), size_raw.width * size_raw.height);

  input = tmp.cast<float>() / 255.0;
}

float sigmoid(float x)
{
  return 1.0 / (1 + exp(-x));
}

void sigmoid(Eigen::Matrix<float, 10, 1>& in, Eigen::Matrix<float, 10, 1>& out)
{
  for (uint16_t i = 0; i < 10; i++)
  {
    out[i] = sigmoid(in[i]);
  }
}

int main(int, char**)
{
  cout << "Hello, world!" << endl;

  std::ifstream dataset_train_images(PROJECT_PATH + std::string("/dataset/train-images-idx3-ubyte/train-images.idx3-ubyte"), std::ios::binary);
  std::ifstream dataset_train_labels(PROJECT_PATH + std::string("/dataset/train-labels-idx1-ubyte/train-labels.idx1-ubyte"), std::ios::binary);
  std::vector<char> images_bytes((std::istreambuf_iterator<char>(dataset_train_images)), std::istreambuf_iterator<char>());
  std::vector<char> labels_bytes((std::istreambuf_iterator<char>(dataset_train_labels)), std::istreambuf_iterator<char>());

  size_t image_counter = 0;

  cv::Mat image(size_raw, CV_8UC1, cv::Scalar(0));
  cv::Mat image_big;

  w2.setRandom();
  w3.setRandom();

  uint8_t label = 0;

  while (1)
  {
    getImage(images_bytes.data(), image_counter, image);
    getLabel(labels_bytes.data(), image_counter, label);
    cout << "Label: " << (int)label << endl;

    getInput(images_bytes.data(), image_counter, a1);

    z2 = w2 * a1 + b2;
    sigmoid(z2, a2);
    z3 = w3 * a2 + b3;
    sigmoid(z3, a3);

    cout << a3 << endl;

    cv::resize(image, image_big, size_big);

    cv::cvtColor(image_big, image_big, cv::COLOR_GRAY2BGR);
    cv::putText(image_big, std::to_string((int)image_counter), cv::Point(10, 30), cv::FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(118, 255, 0), 2);
    cv::putText(image_big, std::to_string((int)label), cv::Point(10, 60), cv::FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(118, 255, 0), 2);

    cv::imshow("Image", image_big);
    char c = cv::waitKey();

    if (c == 27)
    {
      break;
    }

    image_counter++;
  }

  dataset_train_images.close();
  dataset_train_labels.close();

  return 0;
}
