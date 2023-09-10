#include <fstream>
#include <iostream>
#include <iterator>
#include <thread>
#include <vector>

#include <eigen3/Eigen/Eigen>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>

#include "include/project_path.h"

using std::cout;
using std::endl;

// Data set image size: 28x28 pixels, total 784
// sigmoid 1/(1+e^(-x))
// Eigen matrix A(row, col)
// training set label file: 32 bit 2049 (magic number), 32 bit 60000 (number of items), 8 bit label...
// training set image file: 32 bit 2051 (magic number), 32 bit 60000 (number of items), 32 bit 28 (number of rows), 32 bit 28 (number of cols), 8 bit pixel...

const cv::Size size_raw(28, 28);   // original one number
const cv::Size size_big(250, 250); // upscaled one number
// const cv::Size size_raw(28, 28 * 10);   // original row of numbers
// const cv::Size size_big(100, 100 * 10); // upscaled row of numbers
// const cv::Size size_raw(28 * 3, 28 * 1);   // original matrix of numbers
// const cv::Size size_big(100 * 3, 100 * 1); // upscaled matrix of numbers

const uint8_t num_neurons = 10;
// const uint8_t num_threads = 1; // 275 ms epoch
// const uint8_t num_threads = 2; // 150 ms
// const uint8_t num_threads = 3; // 100 ms
// const uint8_t num_threads = 6; // 70 ms
const uint8_t num_threads = 10; // 70 ms

std::thread th[num_threads];

typedef Eigen::Matrix<float, num_neurons, 1> Vector;
typedef Eigen::Matrix<float, 10, 1> Vec10f;

Eigen::Matrix<float, 28 * 28, 1> a1;               // input layer
Vector a2;                                         // hidden layer
Vector a3;                                         // output layer
Vector z2;                                         //
Vector z3;                                         //
Eigen::Matrix<float, num_neurons, 28 * 28> w2;     // 2 layer weight
Eigen::Matrix<float, num_neurons, num_neurons> w3; // 3 layer weight
Vector b2;                                         // 2 layer bias
Vector b3;                                         // 3 layer bias
Vector be2;                                        // 2 layer error
Vector be3;                                        // 3 layer error

Eigen::Matrix<float, num_neurons, 28 * 28> dC_dw2;     //
Eigen::Matrix<float, num_neurons, num_neurons> dC_dw3; //
Vector dC_db2;                                         //
Vector dC_db3;                                         //

Eigen::Matrix<float, num_neurons, 28 * 28> dC_dw2_avr[num_threads];     //
Eigen::Matrix<float, num_neurons, num_neurons> dC_dw3_avr[num_threads]; //
Vector dC_db2_avr[num_threads];                                         //
Vector dC_db3_avr[num_threads];                                         //

size_t image_counter = 0;
const uint32_t number_of_images = 60000;
uint8_t label = 0;

float mu = 0.01;
uint16_t epoch = 0;
float C_train[num_threads];
float C_all = 0;

std::mutex m1;
std::mutex m2;

void testTh(float a)
{
  a = 10;
}

void getImage(char* file_array, size_t image_number, cv::Mat& image)
{
  memcpy(image.data, file_array + 4 * sizeof(uint32_t) + image_number * size_raw.area(), size_raw.width * size_raw.height);
}

void getLabel(char* file_array, size_t image_number, uint8_t& label)
{
  memcpy(&label, file_array + 2 * sizeof(uint32_t) + image_number, 1);
}

uint8_t getAnswer(Vector& nn_out)
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

Vec10f getDesiredOutput(uint8_t label)
{
  // m2.lock();

  Vec10f y;
  y.setZero();

  y[label] = 1;

  // m2.unlock();

  return y;
}

float sigmoid(float x)
{
  return 1.0 / (1 + exp(-x));
}

Vector sigmoid(Vector& in)
{
  // m1.lock();

  Vector out;

  for (uint16_t i = 0; i < 10; i++)
  {
    out[i] = sigmoid(in[i]);
  }

  // m1.unlock();

  return out;
}

void train(uint8_t thread_num, char* train_images_bytes, char* train_labels_bytes, uint32_t start, uint32_t stop)
{
  C_train[thread_num] = 0;

  dC_dw2_avr[thread_num].setZero();
  dC_dw3_avr[thread_num].setZero();
  dC_db2_avr[thread_num].setZero();
  dC_db3_avr[thread_num].setZero();

  Eigen::Matrix<float, num_neurons, 28 * 28> _dC_dw2;     //
  Eigen::Matrix<float, num_neurons, num_neurons> _dC_dw3; //
  Vector _dC_db2;                                         //
  Vector _dC_db3;                                         //

  uint32_t counter = start;

  while (counter < stop)
  {
    getLabel(train_labels_bytes, counter, label);
    getInput(train_images_bytes, counter, a1);

    z2 = w2 * a1 + b2;
    a2 = sigmoid(z2);
    z3 = w3 * a2 + b3;
    a3 = sigmoid(z3);

    Vector y = getDesiredOutput(label);

    float C = 1.0 / 2.0 * (y - a3).transpose() * (y - a3);

    C_train[thread_num] += C;

    // cout << "C: " << C << endl;
    // cout << "thread[" << (int)thread_num << "] image: " << counter << endl;

    be3 = (y - a3).cwiseProduct(sigmoid(z3).cwiseProduct((Vector::Ones() - sigmoid(z3))));
    be2 = (w3.transpose() * be3).cwiseProduct(sigmoid(z3).cwiseProduct((Vector::Ones() - sigmoid(z3))));

    for (uint16_t j = 0; j < 10; j++)
    {
      _dC_db3(j) = be3(j);

      for (uint16_t k = 0; k < 10; k++)
      {
        _dC_dw3(j, k) = a2(k) * be3[j];
      }
    }

    for (uint16_t j = 0; j < 10; j++)
    {
      _dC_db2(j) = be2(j);

      for (uint16_t k = 0; k < 28 * 28; k++)
      {
        _dC_dw2(j, k) = a1(k) * be2[j];
      }
    }

    dC_dw2_avr[thread_num] += _dC_dw2;
    dC_dw3_avr[thread_num] += _dC_dw3;
    dC_db2_avr[thread_num] += _dC_db2;
    dC_db3_avr[thread_num] += _dC_db3;

    counter++;
  }
}

void saveWeights()
{
  // w2.setRandom();
  // w3.setRandom();
  // b2.setRandom();
  // b3.setRandom();
  // cout << "w2: " << w2 << endl;
  // cout << "w3: " << w3 << endl;
  // cout << "b2: " << b2 << endl;
  // cout << "b3: " << b3 << endl;

  cout << "SAVE WEIGHTS" << endl;

  uint8_t* byte_array_w2 = (uint8_t*)w2.data();
  uint8_t* byte_array_w3 = (uint8_t*)w3.data();
  uint8_t* byte_array_b2 = (uint8_t*)b2.data();
  uint8_t* byte_array_b3 = (uint8_t*)b3.data();

  std::ofstream file_w2("w2.bin", std::ios::binary);
  if (!file_w2)
  {
    cout << "Failed to open file for writing." << endl;
    return;
  }
  file_w2.write(reinterpret_cast<const char*>(byte_array_w2), sizeof(w2));
  file_w2.close();

  std::ofstream file_w3("w3.bin", std::ios::binary);
  if (!file_w3)
  {
    cout << "Failed to open file for writing." << endl;
    return;
  }
  file_w3.write(reinterpret_cast<const char*>(byte_array_w3), sizeof(w3));
  file_w3.close();

  std::ofstream file_b2("b2.bin", std::ios::binary);
  if (!file_b2)
  {
    cout << "Failed to open file for writing." << endl;
    return;
  }
  file_b2.write(reinterpret_cast<const char*>(byte_array_b2), sizeof(b2));
  file_b2.close();

  std::ofstream file_b3("b3.bin", std::ios::binary);
  if (!file_b3)
  {
    cout << "Failed to open file for writing." << endl;
    return;
  }
  file_b3.write(reinterpret_cast<const char*>(byte_array_b3), sizeof(b3));
  file_b3.close();
}

void loadWeights()
{
  cout << "LOAD WEIGHTS" << endl;
  unsigned char read_bytes_w2[sizeof(w2)];
  unsigned char read_bytes_w3[sizeof(w3)];
  unsigned char read_bytes_b2[sizeof(b2)];
  unsigned char read_bytes_b3[sizeof(b3)];

  std::ifstream file_w2("w2.bin", std::ios::binary);
  if (!file_w2)
  {
    cout << "Failed to open file for reading." << endl;
    return;
  }
  file_w2.read(reinterpret_cast<char*>(read_bytes_w2), sizeof(w2));
  file_w2.close();
  memcpy(w2.data(), read_bytes_w2, sizeof(w2));

  std::ifstream file_w3("w3.bin", std::ios::binary);
  if (!file_w3)
  {
    cout << "Failed to open file for reading." << endl;
    return;
  }
  file_w3.read(reinterpret_cast<char*>(read_bytes_w3), sizeof(w3));
  file_w3.close();
  memcpy(w3.data(), read_bytes_w3, sizeof(w3));

  std::ifstream file_b2("b2.bin", std::ios::binary);
  if (!file_b2)
  {
    cout << "Failed to open file for reading." << endl;
    return;
  }
  file_b2.read(reinterpret_cast<char*>(read_bytes_b2), sizeof(b2));
  file_b2.close();
  memcpy(b2.data(), read_bytes_b2, sizeof(b2));

  std::ifstream file_b3("b3.bin", std::ios::binary);
  if (!file_b3)
  {
    cout << "Failed to open file for reading." << endl;
    return;
  }
  file_b3.read(reinterpret_cast<char*>(read_bytes_b3), sizeof(b3));
  file_b3.close();
  memcpy(b3.data(), read_bytes_b3, sizeof(b3));

  // cout << "Read w2 : " << w2 << endl;
  // cout << "Read w3 : " << w3 << endl;
  // cout << "Read b2 : " << b2 << endl;
  // cout << "Read b3 : " << b3 << endl;
}

int main(int, char**)
{
  cout << "Hello, world!" << endl;

  cv::Mat hello(cv::Size(100, 100), CV_8UC1, cv::Scalar(255));
  cv::imshow("hello", hello);
  cv::waitKey();

  std::ifstream dataset_train_images(PROJECT_PATH + std::string("/dataset/train-images-idx3-ubyte/train-images.idx3-ubyte"), std::ios::binary);
  std::ifstream dataset_train_labels(PROJECT_PATH + std::string("/dataset/train-labels-idx1-ubyte/train-labels.idx1-ubyte"), std::ios::binary);
  std::vector<char> train_images_bytes((std::istreambuf_iterator<char>(dataset_train_images)), std::istreambuf_iterator<char>());
  std::vector<char> train_labels_bytes((std::istreambuf_iterator<char>(dataset_train_labels)), std::istreambuf_iterator<char>());

  cv::Mat image(size_raw, CV_8UC1, cv::Scalar(0));
  cv::Mat image_big;

  w2.setRandom();
  w3.setRandom();
  b2.setRandom();
  b3.setRandom();

  getImage(train_images_bytes.data(), image_counter, image);
  getLabel(train_labels_bytes.data(), image_counter, label);
  getInput(train_images_bytes.data(), image_counter, a1);

  loadWeights();

  for (uint16_t i = 0; i < 10000; i++)
  {
    auto start = std::chrono::high_resolution_clock::now();
    epoch++;
    cout << "Start training, epoch: " << epoch << endl;

    // train(0, train_images_bytes.data(), train_labels_bytes.data(), 0);

    for (uint16_t i = 0; i < num_threads; i++)
    {
      th[i] = std::thread(train, i, train_images_bytes.data(), train_labels_bytes.data(), number_of_images / num_threads * i, number_of_images / num_threads * (i + 1));
    }

    for (uint16_t i = 0; i < num_threads; i++)
    {
      th[i].join();
    }

    C_all = 0;
    dC_dw2.setZero();
    dC_dw3.setZero();
    dC_db2.setZero();
    dC_db3.setZero();

    for (uint16_t thread = 0; thread < num_threads; thread++)
    {
      C_all += C_train[thread];
      dC_dw2 += dC_dw2_avr[thread];
      dC_dw3 += dC_dw3_avr[thread];
      dC_db2 += dC_db2_avr[thread];
      dC_db3 += dC_db3_avr[thread];
    }

    C_all = C_all / (float)number_of_images;
    dC_dw2 = dC_dw2 / (float)number_of_images;
    dC_dw3 = dC_dw3 / (float)number_of_images;
    dC_db2 = dC_db2 / (float)number_of_images;
    dC_db3 = dC_db3 / (float)number_of_images;

    w2 = w2 + mu * dC_dw2;
    w3 = w3 + mu * dC_dw3;
    b2 = b2 + mu * dC_db2;
    b3 = b3 + mu * dC_db3;

    auto stop = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    cout << "time: " << time.count() / 1000.0 << " ms" << endl;

    cout << "Epoch " << epoch << " training ended, C avr: " << C_all << endl;
  }

  saveWeights();

  image_counter = 0;

  while (1)
  {
    getImage(train_images_bytes.data(), image_counter, image);
    getLabel(train_labels_bytes.data(), image_counter, label);
    getInput(train_images_bytes.data(), image_counter, a1);

    z2 = w2 * a1 + b2;
    a2 = sigmoid(z2);
    z3 = w3 * a2 + b3;
    a3 = sigmoid(z3);

    int answer = getAnswer(a3);

    cv::resize(image, image_big, size_big);

    cv::cvtColor(image_big, image_big, cv::COLOR_GRAY2BGR);
    cv::putText(image_big, std::to_string((int)image_counter), cv::Point(10, 30), cv::FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(118, 255, 0), 2);
    cv::putText(image_big, std::to_string((int)label), cv::Point(10, 60), cv::FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(118, 255, 0), 2);
    cv::putText(image_big, std::to_string((int)answer), cv::Point(10, 90), cv::FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(118, 255, 0), 2);

    cv::imshow("Image", image_big);
    char c = cv::waitKey();

    // cout << (int)c << endl;

    if (c == 83)
    {
      getImage(train_images_bytes.data(), image_counter, image);
      getLabel(train_labels_bytes.data(), image_counter, label);
      getInput(train_images_bytes.data(), image_counter, a1);
      image_counter++;
    }

    if (c == 27)
    {
      break;
    }
  }

  dataset_train_images.close();
  dataset_train_labels.close();

  return 0;
}
