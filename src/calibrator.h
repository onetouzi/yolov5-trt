/*
 * @Author: Touzi
 * @Email: wwwtouzi@outlook.com
 * @Description: INT8 量化校准器，虽然继承自 TensorRT 的 `IInt8EntropyCalibrator2`，但实现是项目自定义的
 * @Date: 2026-05-30 15:04:55
 */
#pragma once

#include "macros.h"
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

// 使图像按照给定的宽高去缩放
cv::Mat preprocess_img(cv::Mat &img, int input_w, int input_h);

class Int8EntropyCalibrator2 : public nvinfer1::IInt8EntropyCalibrator2
{
public:
    // 构造函数
    Int8EntropyCalibrator2(int batchsize, int input_w, int input_h, const char *img_dir,
                           const char *calib_table_name, const char *input_blob_name, bool read_cache = true);
    // 析构函数
    virtual ~Int8EntropyCalibrator2();
    // 获取batchsize
    int getBatchSize() const TRT_NOEXCEPT override;
    // 获取每一个batch的数据，从cpu拷贝到gpu
    bool getBatch(void *bindings[], const char *names[], int nbBindings) TRT_NOEXCEPT override;
    // 读取校准数据并保存到缓存中（一般转换精度的时候用）
    const void *readCalibrationCache(size_t &length) TRT_NOEXCEPT override;
    // 首次得到校准数据的时候，将数据写入到硬盘文件中保存起来
    void writeCalibrationCache(const void *cache, size_t length) TRT_NOEXCEPT override;

private:
    int batchsize_;
    int input_w_;                        // 输入图像的宽
    int input_h_;                        // 输入图像的高
    int img_idx_;                        // 遍历图像序号
    std::string img_dir_;                // 图像根目录
    std::vector<std::string> img_files_; // 图像路径列表
    size_t input_count_;                 // 开辟的空间的大小
    std::string calib_table_name_;
    const char *input_blob_name_;
    bool read_cache_;
    void *device_input_; // 指向开辟的空间
    std::vector<char> calib_cache_;
};