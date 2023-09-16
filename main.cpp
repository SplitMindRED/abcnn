#include <csignal>
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

namespace
{
volatile __sig_atomic_t signal_status;
}

void signal_handler(int signal)
{
  signal_status = signal;
  cout << "CUSTOM SIGINT" << endl;
}

const cv::Size size_raw(28, 28);   // original one number
const cv::Size size_big(250, 250); // upscaled one number

// double mu = 0.01;
// double mu = 0.5;
// double mu = 5;
// double mu = 10;
double mu = 40;
const uint8_t num_neurons = 10;
// const uint8_t num_threads = 1; // 275 ms epoch
// const uint8_t num_threads = 2; // 150 ms
// const uint8_t num_threads = 3; // 100 ms
// const uint8_t num_threads = 4; // 100 ms
// const uint8_t num_threads = 6; // 70 ms
// const uint8_t num_threads = 10; // 70 ms
const uint8_t num_threads = 12; // 70 ms

std::thread th[num_threads];

typedef Eigen::Matrix<double, num_neurons, 1> Vector;
typedef Eigen::Matrix<double, 10, 1> Vec10d;

// Eigen::Matrix<double, 28 * 28, 1> a1;               // input layer
// Vector a2;                                         // hidden layer
// Vector a3;                                         // output layer
// Vector z2;                                         //
// Vector z3;                                         //
Eigen::Matrix<double, num_neurons, 28 * 28> w2;     // 2 layer weight
Eigen::Matrix<double, num_neurons, num_neurons> w3; // 3 layer weight
Vector b2;                                          // 2 layer bias
Vector b3;                                          // 3 layer bias
// Vector be2;                                        // 2 layer error
// Vector be3;                                        // 3 layer error

Eigen::Matrix<double, 28 * 28, 1> a1[num_threads]; // input layer
Vector a2[num_threads];                            // hidden layer
Vector a3[num_threads];                            // output layer
Vector z2[num_threads];                            //
Vector z3[num_threads];                            //
// Eigen::Matrix<double, num_neurons, 28 * 28> w2[num_threads];     // 2 layer weight
// Eigen::Matrix<double, num_neurons, num_neurons> w3[num_threads]; // 3 layer weight
// Vector b2[num_threads];                                         // 2 layer bias
// Vector b3[num_threads];                                         // 3 layer bias
Vector be2[num_threads]; // 2 layer error
Vector be3[num_threads]; // 3 layer error

Eigen::Matrix<double, num_neurons, 28 * 28> dC_dw2;     //
Eigen::Matrix<double, num_neurons, num_neurons> dC_dw3; //
Vector dC_db2;                                          //
Vector dC_db3;                                          //

Eigen::Matrix<double, num_neurons, 28 * 28> dC_dw2_avr[num_threads];     //
Eigen::Matrix<double, num_neurons, num_neurons> dC_dw3_avr[num_threads]; //
Vector dC_db2_avr[num_threads];                                          //
Vector dC_db3_avr[num_threads];                                          //

Eigen::Matrix<double, num_neurons, 28 * 28> _dC_dw2[num_threads];     //
Eigen::Matrix<double, num_neurons, num_neurons> _dC_dw3[num_threads]; //
Vector _dC_db2[num_threads];                                          //
Vector _dC_db3[num_threads];                                          //

size_t image_counter = 0;
const uint32_t number_of_images = 60000;
uint8_t label[num_threads];

uint16_t epoch = 0;
double C_train[num_threads];
double C_all = 0;

Vector y[num_threads];
double C[num_threads];

uint32_t counter[num_threads];

std::mutex m1;
std::mutex m2;
std::mutex m3;
std::mutex m4;
std::mutex m5;

void getImage(char* file_array, size_t image_number, cv::Mat& image)
{
  memcpy(image.data, file_array + 4 * sizeof(uint32_t) + image_number * size_raw.area(), size_raw.width * size_raw.height);
}

void getLabel(char* file_array, size_t image_number, uint8_t& label)
{
  // m4.lock();
  memcpy(&label, file_array + 2 * sizeof(uint32_t) + image_number, 1);
  // m4.unlock();
}

uint8_t getAnswer(Vector& nn_out)
{
  // m3.lock();

  uint8_t answer = 0;

  double tmp = 0;

  for (uint16_t i = 0; i < 10; i++)
  {
    if (nn_out(i) > tmp)
    {
      tmp = nn_out[i];
      answer = i;
    }
  }

  // m3.unlock();

  return answer;
}

void getInput(char* file_array, size_t image_number, Eigen::Matrix<double, 28 * 28, 1>& input)
{
  // m5.lock();

  Eigen::Matrix<uint8_t, 28 * 28, 1> tmp;

  memcpy(tmp.data(), file_array + 4 * sizeof(uint32_t) + image_number * size_raw.area(), size_raw.width * size_raw.height);

  input = tmp.cast<double>() / 255.0;

  // m5.unlock();
}

Vec10d getDesiredOutput(uint8_t label)
{
  // m2.lock();

  Vec10d y;
  y.setZero();

  y[label] = 1;

  // m2.unlock();

  return y;
}

double sigmoid(double x)
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

Vec10d error[num_threads];
Vec10d sigmoid_prime[num_threads];

void train(uint8_t thread_num, char* train_images_bytes, char* train_labels_bytes, uint32_t start, uint32_t stop)
{
  C_train[thread_num] = 0;

  dC_dw2_avr[thread_num].setZero();
  dC_dw3_avr[thread_num].setZero();
  dC_db2_avr[thread_num].setZero();
  dC_db3_avr[thread_num].setZero();

  counter[thread_num] = start;

  while (counter[thread_num] < stop)
  {
    getLabel(train_labels_bytes, counter[thread_num], label[thread_num]);
    getInput(train_images_bytes, counter[thread_num], a1[thread_num]);

    z2[thread_num] = w2 * a1[thread_num] + b2;
    a2[thread_num] = sigmoid(z2[thread_num]);
    z3[thread_num] = w3 * a2[thread_num] + b3;
    a3[thread_num] = sigmoid(z3[thread_num]);

    y[thread_num] = getDesiredOutput(label[thread_num]);
    error[thread_num] = y[thread_num] - a3[thread_num];
    // C[thread_num] = 1.0 / 2.0 * (y[thread_num] - a3[thread_num]).transpose() * (y[thread_num] - a3[thread_num]);
    C[thread_num] = 1.0 / 2.0 * (error[thread_num]).transpose() * (error[thread_num]);

    C_train[thread_num] += C[thread_num];

    // cout << "C: " << C << endl;
    // cout << "thread[" << (int)thread_num << "] image: " << counter << endl;

    sigmoid_prime[thread_num] = sigmoid(z3[thread_num]).cwiseProduct((Vector::Ones() - sigmoid(z3[thread_num])));

    be3[thread_num] = (error[thread_num]).cwiseProduct(sigmoid_prime[thread_num]);
    be2[thread_num] = (w3.transpose() * be3[thread_num]).cwiseProduct(sigmoid_prime[thread_num]);

    for (uint16_t j = 0; j < 10; j++)
    {
      _dC_db3[thread_num](j) = be3[thread_num](j);

      for (uint16_t k = 0; k < 10; k++)
      {
        _dC_dw3[thread_num](j, k) = a2[thread_num](k) * be3[thread_num][j];
      }
    }

    for (uint16_t j = 0; j < 10; j++)
    {
      _dC_db2[thread_num](j) = be2[thread_num](j);

      for (uint16_t k = 0; k < 28 * 28; k++)
      {
        _dC_dw2[thread_num](j, k) = a1[thread_num](k) * be2[thread_num][j];
      }
    }

    dC_dw2_avr[thread_num] += _dC_dw2[thread_num];
    dC_dw3_avr[thread_num] += _dC_dw3[thread_num];
    dC_db2_avr[thread_num] += _dC_db2[thread_num];
    dC_db3_avr[thread_num] += _dC_db3[thread_num];

    counter[thread_num]++;
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

  std::signal(SIGINT, signal_handler);

  cv::Mat hello(cv::Size(100, 100), CV_8UC1, cv::Scalar(255));
  cv::imshow("hello", hello);
  cv::waitKey();

  std::ifstream dataset_train_images(PROJECT_PATH + std::string("/dataset/train-images-idx3-ubyte/train-images.idx3-ubyte"), std::ios::binary);
  std::ifstream dataset_train_labels(PROJECT_PATH + std::string("/dataset/train-labels-idx1-ubyte/train-labels.idx1-ubyte"), std::ios::binary);
  std::ifstream dataset_test_images(PROJECT_PATH + std::string("/dataset/t10k-images-idx3-ubyte/t10k-images.idx3-ubyte"), std::ios::binary);
  std::ifstream dataset_test_labels(PROJECT_PATH + std::string("/dataset/t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte"), std::ios::binary);
  std::vector<char> train_images_bytes((std::istreambuf_iterator<char>(dataset_train_images)), std::istreambuf_iterator<char>());
  std::vector<char> train_labels_bytes((std::istreambuf_iterator<char>(dataset_train_labels)), std::istreambuf_iterator<char>());
  std::vector<char> test_images_bytes((std::istreambuf_iterator<char>(dataset_test_images)), std::istreambuf_iterator<char>());
  std::vector<char> test_labels_bytes((std::istreambuf_iterator<char>(dataset_test_labels)), std::istreambuf_iterator<char>());

  cv::Mat image(size_raw, CV_8UC1, cv::Scalar(0));
  cv::Mat image_big;

  w2.setRandom();
  w3.setRandom();
  b2.setRandom();
  b3.setRandom();

  loadWeights();

  for (uint32_t i = 0; i < 600000; i++)
  {
    if (signal_status == 2)
    {
      cout << "STOP TRAINING" << endl;
      break;
    }

    auto start = std::chrono::high_resolution_clock::now();
    epoch++;
    cout << "Start training, epoch: " << epoch << endl;

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

    C_all = C_all / (double)number_of_images;
    dC_dw2 = dC_dw2 / (double)number_of_images;
    dC_dw3 = dC_dw3 / (double)number_of_images;
    dC_db2 = dC_db2 / (double)number_of_images;
    dC_db3 = dC_db3 / (double)number_of_images;

    w2 = w2 + mu * dC_dw2;
    w3 = w3 + mu * dC_dw3;
    b2 = b2 + mu * dC_db2;
    b3 = b3 + mu * dC_db3;

    auto stop = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    cout << "time: " << time.count() / 1000.0 << " ms" << endl;

    cout << "Epoch " << epoch << " training ended, C avr: " << C_all << endl;
  }

  // saveWeights();

  image_counter = 0;

  float percent = 0;
  unsigned long positive = 0;
  unsigned long negative = 0;

  while (signal_status != 2)
  {
    getLabel(train_labels_bytes.data(), image_counter, label[0]);
    getInput(train_images_bytes.data(), image_counter, a1[0]);

    z2[0] = w2 * a1[0] + b2;
    a2[0] = sigmoid(z2[0]);
    z3[0] = w3 * a2[0] + b3;
    a3[0] = sigmoid(z3[0]);

    int answer = getAnswer(a3[0]);

    if (answer == label[0])
    {
      positive++;
    }
    else
    {
      negative++;
    }
    // cout << "positive: " << positive << " negative: " << negative << endl;

    image_counter++;

    if (image_counter >= 60000)
    {
      break;
    }
  }

  cout << "pos + neg: " << positive + negative << endl;
  percent = positive / (float)number_of_images;
  cout << "train percent: " << percent << endl;

  image_counter = 0;
  positive = 0;
  negative = 0;

  while (signal_status != 2)
  {
    getLabel(test_labels_bytes.data(), image_counter, label[0]);
    getInput(test_images_bytes.data(), image_counter, a1[0]);

    z2[0] = w2 * a1[0] + b2;
    a2[0] = sigmoid(z2[0]);
    z3[0] = w3 * a2[0] + b3;
    a3[0] = sigmoid(z3[0]);

    int answer = getAnswer(a3[0]);

    if (answer == label[0])
    {
      positive++;
    }
    else
    {
      negative++;
    }
    // cout << "positive: " << positive << " negative: " << negative << endl;

    image_counter++;

    if (image_counter >= 10000)
    {
      break;
    }
  }

  cout << "pos + neg: " << positive + negative << endl;
  percent = positive / (float)10000;
  cout << "test percent: " << percent << endl;

  image_counter = 0;

  // Validate
  while (signal_status != 2)
  {
    getImage(train_images_bytes.data(), image_counter, image);
    getLabel(train_labels_bytes.data(), image_counter, label[0]);
    getInput(train_images_bytes.data(), image_counter, a1[0]);

    z2[0] = w2 * a1[0] + b2;
    a2[0] = sigmoid(z2[0]);
    z3[0] = w3 * a2[0] + b3;
    a3[0] = sigmoid(z3[0]);

    int answer = getAnswer(a3[0]);

    cv::resize(image, image_big, size_big);

    cv::cvtColor(image_big, image_big, cv::COLOR_GRAY2BGR);
    cv::putText(image_big, std::to_string((int)image_counter), cv::Point(10, 30), cv::FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(118, 255, 0), 2);
    cv::putText(image_big, std::to_string((int)label[0]), cv::Point(10, 60), cv::FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(118, 255, 0), 2);
    cv::putText(image_big, std::to_string((int)answer), cv::Point(10, 90), cv::FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(118, 255, 0), 2);

    cv::imshow("Image", image_big);
    char c = cv::waitKey();

    // cout << (int)c << endl;

    if (c == 83)
    {
      getImage(train_images_bytes.data(), image_counter, image);
      getLabel(train_labels_bytes.data(), image_counter, label[0]);
      getInput(train_images_bytes.data(), image_counter, a1[0]);
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
