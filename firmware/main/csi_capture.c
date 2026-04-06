// ============================================================
// ESP32-S3 CSI Node — csi_capture.c
// Parses incoming CSI callback data into a compact frame
// and pushes it to the inter-task queue.
// ============================================================

#include "csi_capture.h"
#include "esp_log.h"
#include "esp_timer.h"
#include <string.h>

static const char *TAG = "csi_capture";

// Called by the Wi-Fi driver on every received 802.11 frame.
// NOTE: This runs in the **Wi-Fi task context** (not a hardware ISR),
// so we use xQueueSend (not xQueueSendFromISR). IRAM_ATTR is kept
// because ESP-IDF still requires the callback to be in IRAM.
void IRAM_ATTR csi_capture_callback(void *ctx, wifi_csi_info_t *info)
{
    if (!info || !info->buf || !ctx) return;

    QueueHandle_t queue = (QueueHandle_t)ctx;

    csi_frame_t frame = {0};
    frame.magic     = CSI_FRAME_MAGIC;
    frame.node_id   = (uint16_t)CONFIG_NODE_ID;
    frame.timestamp = (uint64_t)esp_timer_get_time();  // microseconds

    // Number of usable subcarriers: LLTF gives 64 int16 I/Q pairs = 128 values
    int16_t *raw    = (int16_t *)info->buf;
    int      count  = info->len / sizeof(int16_t);   // total int16 elements

    // Guard against malformed data: need at least 2 int16 values for 1 I/Q pair
    if (count < 2) return;

    int      usable = (count > CSI_NUM_SUBCARRIERS * 2)
                        ? CSI_NUM_SUBCARRIERS * 2
                        : count;

    frame.num_subcarriers = (uint16_t)(usable / 2);
    memcpy(frame.iq_data, raw, usable * sizeof(int16_t));

    // Non-blocking push from Wi-Fi task context (not ISR, so use xQueueSend
    // with 0 tick timeout instead of xQueueSendFromISR)
    if (xQueueSend(queue, &frame, 0) != pdTRUE) {
        // Queue full — this is expected under high load; drop silently
    }
}
