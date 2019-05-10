#include <jni.h>
#include <string>
#include <sstream>
#include <fstream>
#include <memory>
#include <vector>
#include <algorithm>
#include <time.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>
#include <torch/csrc/jit/lite/mobile.h>

#define PROTOBUF_USE_DLLS 1
#define CAFFE2_USE_LITE_PROTO 1

#include "classes.h"

#define IMG_H 224
#define IMG_W 224
#define IMG_C 3
#define MAX_DATA_SIZE IMG_H * IMG_W * IMG_C

#define alog(...) __android_log_print(ANDROID_LOG_ERROR, "TORCH", __VA_ARGS__);
#define min(a,b) ((a) > (b)) ? (b) : (a)
#define max(a,b) ((a) > (b)) ? (a) : (b)

static std::string storage_dir;
static std::string debug_file_prefix;
#if 0
static float input_data[MAX_DATA_SIZE];
static std::shared_ptr<torch::jit::script::Module> module;
static at::Tensor input{torch::zeros({1, IMG_C, IMG_H, IMG_W})};
#endif
static int debug_counter;

extern "C" JNIEXPORT void JNICALL
Java_facebook_f8demo_ClassifyCamera_setDebugDirectory(
        JNIEnv* env,
        jobject /* this */,
        jstring directory) {
    const char *dirString = env->GetStringUTFChars(directory, 0);
    storage_dir = std::string(dirString);
    debug_file_prefix = storage_dir + "/" + "debug";
    //std::ofstream out(std::string(dirString) + "/" + "debug.txt");
    env->ReleaseStringUTFChars(directory, dirString);
}

extern "C" JNIEXPORT void JNICALL
Java_facebook_f8demo_ClassifyCamera_initModel(
        JNIEnv* env,
        jobject /* this */,
        jobject assetManager) {
#if 0
    AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);

    std::string filename = "res18.pb";
    std::string data;

    AAsset* asset = AAssetManager_open(mgr, filename.c_str(), AASSET_MODE_STREAMING);
    char buf[1024];
    int nb_read = 0;
    while ((nb_read = AAsset_read(asset, buf, 1024)) > 0) {
        data.append(buf, nb_read);
    }
    AAsset_close(asset);

    std::istringstream input{data};
    module = torch::jit::load(input);

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::ones({1, 3, 224, 224}));
    at::Tensor output = module->forward(inputs).toTensor();

    std::ostringstream out;
    out << "ResNet-18" << std::endl;
    out << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << std::endl;

    return env->NewStringUTF(out.str().c_str());
#endif
    std::ifstream input(storage_dir + "/res18.pb");
    //module = torch::jit::load(input);
    load_model(input);
    allocate_input_buffer(IMG_C, IMG_H, IMG_W);
}

float avg_fps = 0.0;
float total_fps = 0.0;
int iters_fps = 10;

extern "C"
JNIEXPORT jstring JNICALL
Java_facebook_f8demo_ClassifyCamera_classification(
        JNIEnv *env,
        jobject /* this */,
        jint h, jint w, jbyteArray Y, jbyteArray U, jbyteArray V,
        jint rowStride, jint pixelStride,
        jboolean infer_HWC) {
    if (!is_model_loaded() /*!module*/) {
        return env->NewStringUTF("Loading...");
    }
    float* input_data = input_buffer();
    jbyte * Y_data = env->GetByteArrayElements(Y, 0);
    jbyte * U_data = env->GetByteArrayElements(U, 0);
    jbyte * V_data = env->GetByteArrayElements(V, 0);

    auto h_offset = max(0, (h - IMG_H) / 2);
    auto w_offset = max(0, (w - IMG_W) / 2);

    auto iter_h = min(IMG_H, h);
    auto iter_w = min(IMG_W, w);

    for (auto i = 0; i < iter_h; ++i) {
        jbyte* Y_row = &Y_data[(h_offset + i) * rowStride];
        jbyte* U_row = &U_data[(h_offset + i)/2 * rowStride];
        jbyte* V_row = &V_data[(h_offset + i)/2 * rowStride];
        for (auto j = 0; j < iter_w; ++j) {
            char y = Y_row[(w_offset+j)];
            char u = U_row[pixelStride * ((w_offset+j) / 2)];
            char v = V_row[pixelStride * ((w_offset+j) / 2)];

            auto b_i = 2 * IMG_H * IMG_W + j * IMG_W + i;
            auto g_i = 1 * IMG_H * IMG_W + j * IMG_W + i;
            auto r_i = 0 * IMG_H * IMG_W + j * IMG_W + i;

            input_data[r_i] = (float) ((float) min(255., max(0., (float) (y + 1.402 * (v - 128))))) / 255.;
            input_data[g_i] = (float) ((float) min(255., max(0., (float) (y - 0.34414 * (u - 128) - 0.71414 * (v - 128))))) / 255.;
            input_data[b_i] = (float) ((float) min(255., max(0., (float) (y + 1.772 * (u - v))))) / 255.;
        }
    }

    //memcpy(input.data<float>(), input_data, IMG_H * IMG_W * IMG_C * sizeof(float));

#if 0
    {
        alog("Debug: %d", debug_counter);
        if (debug_counter++ % 10 == 0) {
            std::ostringstream filename;
            filename << debug_file_prefix << debug_counter;
            std::ofstream out(filename.str());
            alog("Debug file: %s", filename.str().c_str());
            auto data_ptr = input.data<float>();
            for (int i = 0; i < IMG_H * IMG_W * IMG_C; i++) {
                out << data_ptr[i] << " ";
            }
        }
    }
#endif

    //std::vector<torch::jit::IValue> inputs;
    //inputs.push_back(input);

    time_t start, end;
    time(&start);
    //at::Tensor output = module->forward(inputs).toTensor();
    run_model();
    time(&end);

    float fps = 1 / difftime(end, start);
    total_fps += fps;
    avg_fps = total_fps / iters_fps;
    total_fps -= avg_fps;

    constexpr int k = 5;
    float max[k] = {0};
    int max_index[k] = {0};
    // Find the top-k results manually.
    //auto data = output.data<float>();
    float* data = output_buffer();
    for (auto i = 0; i < 1000; ++i) {
        for (auto j = 0; j < k; ++j) {
            if (data[i] > max[j]) {
                for (auto _j = k - 1; _j > j; --_j) {
                    max[_j] = max[_j - 1];
                    max_index[_j] = max_index[_j - 1];
                }
                max[j] = data[i];
                max_index[j] = i;
                break;
            }
        }
    }

    std::ostringstream stringStream;
    stringStream << avg_fps << " FPS\n";
    //stringStream << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << "\n";
    stringStream << data[0] << " " << data[1] << " " << data[2] << "\n";

    for (auto j = 0; j < k; ++j) {
        stringStream << j << ": " << imagenet_classes[max_index[j]] << " - " << max[j] << "\n";
    }

    return env->NewStringUTF(stringStream.str().c_str());
}
