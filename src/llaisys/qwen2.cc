#include "llaisys/models/qwen2.h"

#include "../models/qwen2/qwen2.hpp"

#include "../utils.hpp"

// 不透明句柄：C 世界只握着指针，内部指向 C++ 模型对象
struct LlaisysQwen2Model {
    llaisys::model::Qwen2Model *model;
};

__C {
    struct LlaisysQwen2Model *llaisysQwen2ModelCreate(
        const LlaisysQwen2Meta *meta,
        llaisysDeviceType_t device,
        int *device_ids,
        int ndevice) {
        try {
            if (meta == nullptr || device_ids == nullptr || ndevice < 1) {
                return nullptr;
            }
            auto *model = new llaisys::model::Qwen2Model(*meta, device, device_ids[0]);
            return new LlaisysQwen2Model{model};
        } catch (...) {
            return nullptr;
        }
    }

    void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model *model) {
        try {
            if (model != nullptr) {
                delete model->model;
                delete model;
            }
        } catch (...) {
            // C 边界不泄漏异常
        }
    }

    struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model *model) {
        try {
            return model->model->weights();
        } catch (...) {
            return nullptr;
        }
    }

    int64_t llaisysQwen2ModelInfer(
        struct LlaisysQwen2Model *model,
        int64_t *token_ids,
        size_t ntoken) {
        try {
            return model->model->infer(token_ids, ntoken);
        } catch (...) {
            return -1;
        }
    }
}
