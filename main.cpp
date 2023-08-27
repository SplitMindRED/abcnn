#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>

#include <eigen3/Eigen/Eigen>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>

using std::cout;
using std::endl;

Eigen::Vector3f a;

// sigmoid 1/(1+e^(-x))

const cv::Size size_raw(28, 28);   // original one number
const cv::Size size_big(250, 250); // upscaled one number
// const cv::Size size_raw(28, 28 * 10);   // original row of numbers
// const cv::Size size_big(100, 100 * 10); // upscaled row of numbers
// const cv::Size size_raw(28 * 3, 28 * 1);   // original matrix of numbers
// const cv::Size size_big(100 * 3, 100 * 1); // upscaled matrix of numbers

const size_t image_size = 28 * 28;

cv::Mat getImage(char* file_array, size_t image_number = 0)
{
  cv::Mat image = cv::Mat(size_raw, CV_8UC1, cv::Scalar(0));

  memcpy(image.data, file_array + 4 * sizeof(uint32_t) + image_number * image_size, size_raw.width * size_raw.height);

  return image;
}

int main(int, char**)
{
  cout << "Hello, world!" << endl;

  std::ifstream input("/workspace/neural_network/dataset/train-images-idx3-ubyte/train-images.idx3-ubyte", std::ios::binary);

  std::vector<char> bytes((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
  char* array = new char[size_raw.width * size_raw.height];

  cv::Mat img = cv::Mat(size_raw, CV_8UC1, cv::Scalar(255));

  memcpy(img.data, bytes.data() + 4 * sizeof(uint32_t), size_raw.width * size_raw.height);

  size_t image_counter = 0;

  while (1)
  {
    cv::Mat image = getImage(bytes.data(), image_counter);
    cv::Mat image_big;

    cv::resize(image, image_big, size_big);

    cv::cvtColor(image_big, image_big, cv::COLOR_GRAY2BGR);
    cv::putText(image_big, std::to_string((int)image_counter), cv::Point(10, 30), cv::FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(118, 255, 0), 2);

    cv::imshow("Image", image_big);
    char c = cv::waitKey();

    if (c == 27)
    {
      break;
    }

    image_counter++;

    // if(image_counter > 60)
  }

  input.close();

  delete[] array;

  return 0;
}
