#ifndef YOLO_H_
#define YOLO_H_

#include <cstring>
#include <opencv2/opencv.hpp>

#import "model.h"

namespace nn {

/// 识别结果
struct Objects {
  float x1;                     ///< 预测框左上角顶点横坐标
  float y1;                     ///< 预测框左上角顶点纵坐标
  float x2;                     ///< 预测框右下角顶点横坐标
  float y2;                     ///< 预测框右下角顶点纵坐标
  float prob;                   ///< 置信度
  int cls;                      ///< 类编号
  std::vector<cv::Point2f> pts; ///< 关键点
};

/// yolov8神经网络类
class Yolo {
public:
  Yolo() = default;
  ~Yolo() = default;

  /**
   * @brief 初始化网络
   * @param model_file 模型文件路径
   * @param num_classes 识别物体的种类数量
   * @param num_points 识别物体的关键点数量
   * @return 是否初始化成功
   */
  bool Initialize(std::string const &model_file, int num_classes,
                          int num_points);

  /**
   * @brief 运行神经网络
   * @param image 需要检测的图片
   * @return 检测到的物体
   */
  std::vector<Objects> Run(cv::Mat image);

private:
  Model *model_;                  ///< 模型指针
  CVPixelBufferRef pixel_buffer_; ///< 网络原始输出

  int num_classes_{}; ///< 识别物体的种类数量
  int num_points_{};  ///< 识别物体的关键点数量

  float box_conf_thresh_{0.55}; ///< 识别框置信度阈值
  int max_nms_{3000};           ///< 非极大值抑制最大值
  float iou_thresh_{0.25};      ///< 交并比最大值

  int input_w_{}; ///< 图片宽度
  int input_h_{}; ///< 图片高度

  float *output_data_{nullptr}; ///< 网络推理后的原始输出

  /**
   * @brief 将图片调整为适合网络输入大小的尺寸
   * @param [out] image 调整的图片
   * @param [out] ro 缩放系数
   * @param [out] dw 值：缩放完后和实际宽度差值的一半
   * @param [out] dh 值：缩放完后和实际高度差值的一半
   */
  void LetterBox(cv::Mat &image, float &ro, float &dw, float &dh);

  /**
   * @brief 处理网络运行后的原始数据
   * @param [out] objs 得到物体数据
   */
  void GetObjects(std::vector<Objects> &objs);

  /**
   * @brief 非极大值抑制合并物体
   * @param [out] objs 需要合并的物体
   */
  void NMS(std::vector<Objects> &objs);
};

} // namespace nn

#endif //  YOLO_H_