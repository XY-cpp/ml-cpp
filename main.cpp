#include "yolo.h"
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <toml.hpp>
int main() {
  auto config = toml::parse("../config.toml");
  auto image = cv::imread(config["image_path"].as_string());
  auto model_path = config["model_path"].as_string();
  auto class_num = config["class_num"].as_integer();
  auto point_num = config["point_num"].as_integer();
  nn::Yolo yolo;
  yolo.Initialize(model_path, class_num, point_num);
  auto objects = yolo.Run(image);
  for (auto object : objects) {
    for (int j = 0; j < 4; j++) {
      cv::line(image, object.pts[j], object.pts[(j + 1) % 4],
               cv::Scalar(0, 192, 0), 1);
    }
  }
  std::cout << objects.size() << std::endl;
  cv::imshow("result", image);
  cv::waitKey();
  return 0;
}