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
// Eigen matrix A(row, col)

const cv::Size size_raw(28, 28);   // original one number
const cv::Size size_big(250, 250); // upscaled one number
// const cv::Size size_raw(28, 28 * 10);   // original row of numbers
// const cv::Size size_big(100, 100 * 10); // upscaled row of numbers
// const cv::Size size_raw(28 * 3, 28 * 1);   // original matrix of numbers
// const cv::Size size_big(100 * 3, 100 * 1); // upscaled matrix of numbers

typedef Eigen::Matrix<float, 10, 1> Vec10f;

Eigen::Matrix<float, 28 * 28, 1> a1;  // input layer
Vec10f a2;                            // hidden layer
Vec10f a3;                            // output layer
Vec10f z2;                            //
Vec10f z3;                            //
Eigen::Matrix<float, 10, 28 * 28> w2; // 2 layer weight
Eigen::Matrix<float, 10, 10> w3;      // 3 layer weight
Vec10f b2;                            // 2 layer bias
Vec10f b3;                            // 3 layer bias
Vec10f be2;                           // 2 layer error
Vec10f be3;                           // 3 layer error

Eigen::Matrix<float, 10, 28 * 28> dC_dw2; //
Eigen::Matrix<float, 10, 10> dC_dw3;      //
Vec10f dC_db2;                            //
Vec10f dC_db3;                            //

void getImage(char* file_array, size_t image_number, cv::Mat& image)
{
  memcpy(image.data, file_array + 4 * sizeof(uint32_t) + image_number * size_raw.area(), size_raw.width * size_raw.height);
}

void getLabel(char* file_array, size_t image_number, uint8_t& label)
{
  memcpy(&label, file_array + 2 * sizeof(uint32_t) + image_number, 1);
}

uint8_t getAnswer(Vec10f& nn_out)
{
  uint8_t answer = 0;

  float tmp = 0;

  for (uint16_t i = 0; i < 10; i++)
  {
    if (nn_out(i) > tmp)
    {
      tmp = nn_out[i];
      answer = i;
    }
  }

  return answer;
}

void getInput(char* file_array, size_t image_number, Eigen::Matrix<float, 28 * 28, 1>& input)
{
  static Eigen::Matrix<uint8_t, 28 * 28, 1> tmp;

  memcpy(tmp.data(), file_array + 4 * sizeof(uint32_t) + image_number * size_raw.area(), size_raw.width * size_raw.height);

  input = tmp.cast<float>() / 255.0;
}

Eigen::Matrix<float, 10, 1> getDesiredOutput(uint8_t label)
{
  Eigen::Matrix<float, 10, 1> y;
  y.setZero();

  y[label] = 1;

  return y;
}

float sigmoid(float x)
{
  return 1.0 / (1 + exp(-x));
}

Vec10f sigmoid(Vec10f& in)
{
  Vec10f out;

  for (uint16_t i = 0; i < 10; i++)
  {
    out[i] = sigmoid(in[i]);
  }

  return out;
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

  getImage(images_bytes.data(), image_counter, image);
  getLabel(labels_bytes.data(), image_counter, label);
  getInput(images_bytes.data(), image_counter, a1);

  float mu = 10;

  while (1)
  {
    z2 = w2 * a1 + b2;
    a2 = sigmoid(z2);
    z3 = w3 * a2 + b3;
    a3 = sigmoid(z3);

    Vec10f y = getDesiredOutput(label);

    float C = 1.0 / 2.0 * (y - a3).transpose() * (y - a3);

    cout << "y: " << y.transpose() << endl;
    cout << "a3: " << a3.transpose() << endl;
    cout << "C: " << C << endl;

    be3 = (y - a3).cwiseProduct(sigmoid(z3).cwiseProduct((Vec10f::Ones() - sigmoid(z3))));
    be2 = (w3.transpose() * be3).cwiseProduct(sigmoid(z3).cwiseProduct((Vec10f::Ones() - sigmoid(z3))));
    // cout << "be3: " << endl << be3 << endl;
    // cout << "be2: " << endl << be2 << endl;

    for (uint16_t j = 0; j < 10; j++)
    {
      dC_db3(j) = be3(j);

      for (uint16_t k = 0; k < 10; k++)
      {
        dC_dw3(j, k) = a2(k) * be3[j];
      }
    }

    for (uint16_t j = 0; j < 10; j++)
    {
      dC_db2(j) = be2(j);

      for (uint16_t k = 0; k < 28 * 28; k++)
      {
        dC_dw2(j, k) = a1(k) * be2[j];
      }
    }

    w2 = w2 + mu * dC_dw2;
    w3 = w3 + mu * dC_dw3;

    b2 = b2 + mu * dC_db2;
    b3 = b3 + mu * dC_db3;

    int answer = getAnswer(a3);

    cv::resize(image, image_big, size_big);

    cv::cvtColor(image_big, image_big, cv::COLOR_GRAY2BGR);
    cv::putText(image_big, std::to_string((int)image_counter), cv::Point(10, 30), cv::FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(118, 255, 0), 2);
    cv::putText(image_big, std::to_string((int)label), cv::Point(10, 60), cv::FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(118, 255, 0), 2);
    cv::putText(image_big, std::to_string((int)answer), cv::Point(10, 90), cv::FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(118, 255, 0), 2);

    cv::imshow("Image", image_big);
    char c = cv::waitKey();
    // cout << (int)c << endl;

    // left arrow - 83

    if (c == 83)
    {
      getImage(images_bytes.data(), image_counter, image);
      getLabel(labels_bytes.data(), image_counter, label);
      getInput(images_bytes.data(), image_counter, a1);
    }

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
