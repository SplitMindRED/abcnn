#include <opencv2/opencv.hpp>

using std::cout;
using std::endl;

// Global variables
cv::Mat image;
bool drawing = false;
cv::Point prevPoint;

const std::string win1 = "Image";
const std::string win2 = "img small";
const std::string win3 = "img big";

// Mouse callback function
void onMouse(int event, int x, int y, int flags, void* userdata)
{
  if (event == cv::EVENT_LBUTTONDOWN)
  {
    drawing = true;
    prevPoint = cv::Point(x, y);
  }
  else if (event == cv::EVENT_MOUSEMOVE)
  {
    if (drawing)
    {
      cv::Point currentPoint(x, y);
      cv::line(image, prevPoint, currentPoint, cv::Scalar(255, 255, 255), 10);
      prevPoint = currentPoint;
      cv::imshow(win1, image);

      cv::Mat img_small;
      cv::Mat img_big;
      cv::resize(image, img_small, cv::Size(28, 28));
      cv::imshow(win2, img_small);
      cv::resize(img_small, img_big, cv::Size(280, 280));
      cv::imshow(win3, img_big);
    }
  }
  else if (event == cv::EVENT_LBUTTONUP)
  {
    drawing = false;
  }
}

int main()
{
  cv::Mat hello(cv::Size(250, 250), CV_8UC1, cv::Scalar(0));
  image = hello.clone();

  if (image.empty())
  {
    std::cerr << "Error: Could not open or find the image!" << std::endl;
    return -1;
  }

  cv::namedWindow(win1);
  cv::namedWindow(win2);
  cv::namedWindow(win3);

  cv::moveWindow(win2, 500, 0);
  cv::moveWindow(win3, 1000, 0);
  cv::imshow(win1, image);

  // Set up the mouse callback function
  cv::setMouseCallback(win1, onMouse, NULL);

  while (true)
  {
    int key = cv::waitKey(10);

    if (key == 'c')
    {
      cout << "clear" << endl;
      image = hello.clone();
      //   image = cv::Mat(cv::Size(250, 250), CV_8UC1, cv::Scalar(0));
      cv::imshow(win1, image);
    }

    if (key == 27) // Exit when 'Esc' key is pressed
    {
      break;
    }
  }

  cv::destroyAllWindows();
  return 0;
}