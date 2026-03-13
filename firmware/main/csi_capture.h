#pragma once
#include <stdint.h>
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "esp_wifi.h"

// Magic number identifying a valid CSI frame
#define CSI_FRAME_MAGIC      0x43534931U  // "CSI1"
#define CSI_NUM_SUBCARRIERS  64

// Compact binary CSI frame — matches aggregator's Rust struct layout
typedef struct __attribute__((packed)) {
    uint32_t magic;
    uint16_t node_id;
    uint64_t timestamp;          // microseconds since ESP boot
    uint16_t num_subcarriers;
    int16_t  iq_data[CSI_NUM_SUBCARRIERS * 2];  // interleaved I, Q
} csi_frame_t;

void IRAM_ATTR csi_capture_callback(void *ctx, wifi_csi_info_t *info);
