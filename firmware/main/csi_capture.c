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
// Runs in Wi-Fi task context — keep it fast, no blocking.
void IRAM_ATTR csi_capture_callback(void *ctx, wifi_csi_info_t *info)
{
    if (!info || !info->buf) return;

    QueueHandle_t queue = (QueueHandle_t)ctx;

    csi_frame_t frame = {0};
    frame.magic     = CSI_FRAME_MAGIC;
    frame.node_id   = (uint16_t)CONFIG_NODE_ID;
    frame.timestamp = (uint64_t)esp_timer_get_time();  // microseconds

    // Number of usable subcarriers: LLTF gives 64 int16 I/Q pairs = 128 values
    int16_t *raw    = (int16_t *)info->buf;
    int      count  = info->len / sizeof(int16_t);   // total int16 elements
    int      usable = (count > CSI_NUM_SUBCARRIERS * 2)
                        ? CSI_NUM_SUBCARRIERS * 2
                        : count;

    frame.num_subcarriers = (uint16_t)(usable / 2);
    memcpy(frame.iq_data, raw, usable * sizeof(int16_t));

    // Non-blocking push; drop frame if queue full (backpressure)
    BaseType_t woken = pdFALSE;
    if (xQueueSendFromISR(queue, &frame, &woken) != pdTRUE) {
        // Queue full — this is expected under high load; drop silently
    }
    portYIELD_FROM_ISR(woken);
}
