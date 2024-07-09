#include "yolo.h"
#include <cstring>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>

namespace nn {

bool Yolo::Initialize(std::string const &model_file, int num_classes,
                      int num_points) {
  num_classes_ = num_classes;
  num_points_ = num_points;

  const std::string &s = model_file;
  int p = s.rfind('.');
  if (s.substr(p + 1) != "mlmodelc") {
    LOG(ERROR) << "The extension of the file should be named as \'.mlmodelc\'";
    return false;
  }

  NSError *error = nil;
  NSString *file = [NSString stringWithCString:s.substr(0, p).c_str()
                                      encoding:NSUTF8StringEncoding];
  NSString *extension = [NSString stringWithCString:s.substr(p + 1).c_str()
                                           encoding:NSUTF8StringEncoding];
  NSURL *model_url = [[NSBundle mainBundle] URLForResource:file
                                             withExtension:extension];

  model_ = [[Model alloc] initWithMLModel:model_url error:&error];
  if (error != nullptr) {
    LOG(ERROR) << "Fail to load model: "
               << [[error localizedDescription] UTF8String];
    return false;
  }

  input_w_ = model_.inputW;
  input_h_ = model_.inputH;

  CVPixelBufferCreate(
      kCFAllocatorDefault, input_w_, input_h_, kCVPixelFormatType_32BGRA,
      (__bridge CFDictionaryRef)
          @{(NSString *)kCVPixelBufferIOSurfacePropertiesKey : @{}},
      &pixel_buffer_);

  return true;
}

std::vector<Objects> Yolo::Run(cv::Mat image) {
  float ro, dw, dh;
  LetterBox(image, ro, dw, dh);

  CVPixelBufferLockBaseAddress(pixel_buffer_, 0);
  void *dest = CVPixelBufferGetBaseAddress(pixel_buffer_);
  size_t bytes_per_row = CVPixelBufferGetBytesPerRow(pixel_buffer_);
  cv::Mat cv_dest(image.rows, image.cols, CV_8UC4, dest, bytes_per_row);
  cv::cvtColor(image, cv_dest, cv::COLOR_BGR2BGRA);
  CVPixelBufferUnlockBaseAddress(pixel_buffer_, 0);

  NSError *error = nil;
  ModelOutput *output_raw = [model_ predictionFromImage:pixel_buffer_
                                                  error:&error];
  if (error != nullptr) {
    LOG(ERROR) << "Failed to get output from model: "
               << [[error localizedDescription] UTF8String];
    return {};
  }

  void *data_pointer = output_raw.out.dataPointer;
  output_data_ = static_cast<float *>(data_pointer);

  std::vector<Objects> objs;
  GetObjects(objs);
  NMS(objs);

  for (auto &[x1, y1, x2, y2, prob, cls, apex] : objs) {
    x1 -= dw, x2 -= dw, y1 -= dh, y2 -= dh;
    x1 /= ro, x2 /= ro, y1 /= ro, y2 /= ro;

    for (auto &[x, y] : apex) {
      x -= dw, y -= dh;
      x /= ro, y /= ro;
    }
  }

  delete[] output_data_;
  return objs;
}

void Yolo::LetterBox(cv::Mat &image, float &ro, float &dw, float &dh) {
  cv::Size shape = image.size();
  cv::Size new_shape = {input_w_, input_h_};
  ro = std::min(new_shape.width / (float)shape.width,
                new_shape.height / (float)shape.height);

  // Compute padding
  cv::Size new_unpad = {(int)round(shape.width * ro),
                        (int)round(shape.height * ro)};
  dw = new_shape.width - new_unpad.width,
  dh = new_shape.height - new_unpad.height; // wh padding

  // divide padding into 2 sides
  dw /= 2.0, dh /= 2.0;

  if (shape != new_unpad) { // resize
    cv::resize(image, image, new_unpad, 0, 0, cv::INTER_LINEAR);
  }

  int top = round(dh - 0.1), bottom = round(dh + 0.1);
  int left = round(dw - 0.1), right = round(dw + 0.1);
  cv::copyMakeBorder(image, image, top, bottom, left, right,
                     cv::BORDER_CONSTANT, {114, 114, 114}); // add border
}

void Yolo::GetObjects(std::vector<Objects> &objs) {
  int n = (input_w_ / 8 * input_h_ / 8 + input_w_ / 16 * input_h_ / 16 +
           input_w_ / 32 * input_h_ / 32);
  int m = (4 + num_classes_ + 2 * num_points_);
  for (int i = 0; i < n; i++) {
    float data[m];
    for (int j = 0; j < m; j++) {
      data[j] = output_data_[j * n + i];
    }

    int cls = std::max_element(data + 4, data + 4 + num_classes_) - (data + 4);
    float prob = data[4 + cls];
    if (prob < box_conf_thresh_) {
      continue;
    }
    Objects obj;
    obj.cls = cls;
    obj.prob = prob;

    float &x = data[0];
    float &y = data[1];
    float &w = data[2];
    float &h = data[3];

    obj.x1 = x - w / 2;
    obj.y1 = y - h / 2;
    obj.x2 = x + w / 2;
    obj.y2 = y + h / 2;

    for (int j = 4 + num_classes_; j < m; j += 2) {
      obj.pts.emplace_back(data[j], data[j + 1]);
    }

    objs.push_back(obj);
  }
}

void Yolo::NMS(std::vector<Objects> &objs) {
  std::sort(objs.begin(), objs.end(),
            [](Objects &a, Objects &b) { return a.prob > b.prob; });
  if (objs.size() > max_nms_) {
    objs.resize(max_nms_);
  }
  std::vector<float> v_area(objs.size());
  for (size_t i = 0; i < objs.size(); i++) {
    v_area[i] = (objs[i].x2 - objs[i].x1 + 1) * (objs[i].y2 - objs[i].y1 + 1);
  }
  for (size_t i = 0; i < objs.size(); i++) {
    for (size_t j = i + 1; j < objs.size();) {
      float xx1 = std::fmax(objs[i].x1, objs[j].x1);
      float yy1 = std::fmax(objs[i].y1, objs[j].y1);
      float xx2 = std::fmin(objs[i].x2, objs[j].x2);
      float yy2 = std::fmin(objs[i].y2, objs[j].y2);
      float w = std::fmax(0, xx2 - xx1 + 1);
      float h = std::fmax(0, yy2 - yy1 + 1);
      float inter = w * h;
      float ovr = inter / (v_area[i] + v_area[j] - inter);
      if (ovr >= iou_thresh_) {
        objs.erase(objs.begin() + j);
        v_area.erase(v_area.begin() + j);
      } else {
        j++;
      }
    }
  }
}

} // namespace nn