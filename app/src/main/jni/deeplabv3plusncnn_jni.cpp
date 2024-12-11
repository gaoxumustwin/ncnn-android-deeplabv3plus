// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <android/asset_manager_jni.h>
#include <android/bitmap.h>
#include <android/log.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <jni.h>
#include <string>
#include <vector>

// ncnn
#include "net.h"
#include "benchmark.h"

static ncnn::Net deeplabv3plusnet;

static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
static ncnn::PoolAllocator g_workspace_pool_allocator;

cv::Mat create_voc_color_map()
{
    int N = 21;
    cv::Mat cmap(N, 1, CV_8UC3);
    for (int i = 0; i < N; i++) {
        int r = 0, g = 0, b = 0;
        int c = i;
        for (int j = 0; j < 8; j++) {
            // opencv 是BGR的色彩空间
            r = r | ((c & (1 << 0)) >> 0) << (7 - j);
            g = g | ((c & (1 << 1)) >> 1) << (7 - j);
            b = b | ((c & (1 << 2)) >> 2) << (7 - j);

            c = c >> 3;
        }

        cmap.at<cv::Vec3b>(i) = cv::Vec3b(r, g, b);
    }
    return cmap;
}


extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "DeepLabv3PlusNcnn", "JNI_OnLoad");

    ncnn::create_gpu_instance();

    return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "DeepLabv3PlusNcnn", "JNI_OnUnload");

    ncnn::destroy_gpu_instance();
}

// public native boolean Init(AssetManager mgr);
JNIEXPORT jboolean JNICALL Java_com_tencent_deeplabv3plusncnn_DeepLabv3PlusNcnn_Init(JNIEnv* env, jobject thiz, jobject assetManager)
{
    ncnn::Option opt;

    opt.lightmode = true;
    opt.num_threads = 4;
    opt.blob_allocator = &g_blob_pool_allocator;
    opt.workspace_allocator = &g_workspace_pool_allocator;
    opt.use_packing_layout = true;

    // use vulkan compute
    if (ncnn::get_gpu_count() != 0)
        opt.use_vulkan_compute = true;

    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);

    deeplabv3plusnet.opt = opt;

    // init param
    {
        int ret = deeplabv3plusnet.load_param(mgr, "deeplabv3+.param");
        if (ret != 0) {
            __android_log_print(ANDROID_LOG_DEBUG, "deeplabv3+", "load_param failed");
            return JNI_FALSE;
        } else {
            __android_log_print(ANDROID_LOG_DEBUG, "deeplabv3+", "load param successful");
        }
    }

    // init bin
    {
        int ret = deeplabv3plusnet.load_model(mgr, "deeplabv3+.bin");
        if (ret != 0) {
            __android_log_print(ANDROID_LOG_DEBUG, "deeplabv3+", "load_model failed");
            return JNI_FALSE;
        } else {
            __android_log_print(ANDROID_LOG_DEBUG, "deeplabv3+", "load bin successful");
        }
    }

    return JNI_TRUE;
}

// public native Bitmap Transfer(Bitmap bitmap, boolean use_gpu);
JNIEXPORT jboolean JNICALL Java_com_tencent_deeplabv3plusncnn_DeepLabv3PlusNcnn_Transfer(JNIEnv* env, jobject thiz, jobject bitmap, jboolean use_gpu)
{
    if (use_gpu == JNI_TRUE && ncnn::get_gpu_count() == 0)
        return JNI_FALSE;

    double start_time = ncnn::get_current_time();

    AndroidBitmapInfo info;
    AndroidBitmap_getInfo(env, bitmap, &info);
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888)
        return JNI_FALSE;

    // ncnn from bitmap
    ncnn::Mat in = ncnn::Mat::from_android_bitmap_resize(env, bitmap, ncnn::Mat::PIXEL_RGB,513 ,513);
    const float mean_vals[3] = {0.485f * 255.f, 0.456f * 255.f, 0.406f * 255.f};
    const float norm_vals[3] = {1 / (0.229f * 255.f), 1 / (0.224f * 255.f), 1 / (0.225f * 255.f)};
    in.substract_mean_normalize(mean_vals, norm_vals);

    // in dimensions: 513 x 513 x 3
    __android_log_print(ANDROID_LOG_DEBUG, "DeepLabv3PlusNcnn", "in dimensions: %d x %d x %d", in.w, in.h, in.c); //  in dimensions: 256 x 256 x 3

    // infer
    ncnn::Mat out;
    {
        ncnn::Extractor ex = deeplabv3plusnet.create_extractor();
        ex.set_vulkan_compute(use_gpu);
        ex.input("images", in);
        ex.extract("output", out);
    }

    // out dimensions: 513 x 513 x 21
    __android_log_print(ANDROID_LOG_DEBUG, "DeepLabv3PlusNcnn", "out dimensions: %d x %d x %d", out.w, out.h, out.c);

    // select the maximum value of each pixel as the category
    std::vector<int> pred(out.h * out.w);
    for (int i = 0; i < out.h * out.w; i++) {
        float max_prob = -1;
        int max_label = 0;
        for (int j = 0; j < out.c; j++) {
            float prob = out.channel(j)[i];
            if (prob > max_prob) {
                max_prob = prob;
                max_label = j;
            }
        }
        pred[i] = max_label;  // 将每个像素的类别索引保存
    }

    // Colorize the processed image
    cv::Mat colorized(out.h, out.w, CV_8UC3);
    cv::Mat cmap = create_voc_color_map();
    for (int i = 0; i < out.h * out.w; i++) {
        int label = pred[i];
//        __android_log_print(ANDROID_LOG_DEBUG, "DeepLabv3PlusNcnn  label", "%d", label);
        if (label >= 0 && label < 21) {
            colorized.at<cv::Vec3b>(i / out.w, i % out.w) = cmap.at<cv::Vec3b>(label);
        } else {
            colorized.at<cv::Vec3b>(i / out.w, i % out.w) = cv::Vec3b(255, 255, 255);
        }
    }

    // cv::Mat to ncnn::Mat
    ncnn::Mat ncnn_colorized(out.w, out.h, 3);  // 3代表 RGB 通道

    for (int y = 0; y < out.h; y++) {
        for (int x = 0; x < out.w; x++) {
            cv::Vec3b pixel = colorized.at<cv::Vec3b>(y, x);
            ncnn_colorized.channel(0)[y * out.w + x] = pixel[0]; // R
            ncnn_colorized.channel(1)[y * out.w + x] = pixel[1]; // G
            ncnn_colorized.channel(2)[y * out.w + x] = pixel[2]; // B
        }
    }

    // ncnn to bitmap
    ncnn_colorized.to_android_bitmap(env, bitmap, ncnn::Mat::PIXEL_RGB);

    double elasped = ncnn::get_current_time() - start_time;
    __android_log_print(ANDROID_LOG_DEBUG, "DeepLabv3PlusNcnn", "%.2fms  transfer", elasped);

    return JNI_TRUE;
}
}