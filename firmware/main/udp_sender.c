// ============================================================
// ESP32-S3 CSI Node — udp_sender.c
// Dequeues CSI frames and streams them as raw binary UDP
// packets to the host aggregator.
// ============================================================

#include "udp_sender.h"
#include "csi_capture.h"
#include "esp_log.h"
#include <string.h>
#include "lwip/sockets.h"
#include "lwip/netdb.h"
#include "freertos/task.h"

static const char *TAG = "udp_sender";

void udp_sender_task(void *arg)
{
    QueueHandle_t queue = (QueueHandle_t)arg;

    // Resolve host address
    struct sockaddr_in dest_addr = {
        .sin_family = AF_INET,
        .sin_port   = htons(CONFIG_HOST_PORT),
    };
    inet_aton(CONFIG_HOST_IP, &dest_addr.sin_addr);

    int sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
    if (sock < 0) {
        ESP_LOGE(TAG, "Failed to create socket");
        vTaskDelete(NULL);
        return;
    }

    // Set send buffer large enough for burst
    int buf_size = 65536;
    setsockopt(sock, SOL_SOCKET, SO_SNDBUF, &buf_size, sizeof(buf_size));

    ESP_LOGI(TAG, "UDP sender ready → %s:%d", CONFIG_HOST_IP, CONFIG_HOST_PORT);

    csi_frame_t frame;
    while (1) {
        if (xQueueReceive(queue, &frame, portMAX_DELAY) == pdTRUE) {
            int sent = sendto(sock, &frame, sizeof(csi_frame_t), 0,
                              (struct sockaddr *)&dest_addr, sizeof(dest_addr));
            if (sent < 0) {
                ESP_LOGW(TAG, "sendto failed: errno %d", errno);
            }
        }
    }

    close(sock);
    vTaskDelete(NULL);
}
