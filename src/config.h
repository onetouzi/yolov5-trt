/*
 * @Author: Touzi
 * @Email: wwwtouzi@outlook.com
 * @Description: tensorrt model 的 config 文件
 * @Date: 2026-05-30 10:56:16
 */
#pragma once

/**
 *这些配置与trt模型有关，如果修改请重新编译，重新序列化trt模型
 */

#define USE_FP32 // set USE_INT8 OR USE_FP16 OR USE_FP32

// 输入/输出 张量的名字
const static char *kInputTensorName = "data";
const static char *kOutputTensorName = "prob";

// 目标检测和分割的类别数量（coco数据集）
constexpr static int kNumClass = 80;

// 分类的类别数量
constexpr static int kClsNumClass = 1000;

// 显式设置batchsize
constexpr static int kBatchSize = 1;

// yolo模型输入的图像的宽高必须是32的倍数
constexpr static int kInputH = 640;
constexpr static int kInputW = 640;

// 分类模型的输入形状
constexpr static int kClsInputH = 224;
constexpr static int kClsInputW = 224;

// 输出的边界框上限
constexpr static int kMaxNumOutputBbox = 1000;

// Anchor数量
constexpr static int kNumAnchor = 3;

// IOU惩罚阈值，IOU超过这个值的就不惩罚
constexpr static float kIgnoreThresh = 0.1;

/**
 *这些配置与trt模型无关，如果修改请重新编译，不用重新序列化trt模型
 */

// NMS IOU 阈值
/**
 * YOLO 会对同一个目标生成多个高度重叠的预测框。NMS 算法会将这些框按置信度排序，
 * 选出得分最高的一个作为基准，然后计算其余框与它的 IoU。如果某个框与最高分框的
 * 重叠度（IoU）大于 0.45，就会被判定为重复框并被“抑制（删除）”；反之则保留。
 */
constexpr static float kNmsThresh = 0.45f;

// 置信度阈值
constexpr static float kConfThresh = 0.5f;

// 绑定的GPU编号
constexpr static int kGpuId = 0;

// 单张图片的最大像素值
constexpr static int kMaxInputImageSize = 4096 * 3112;