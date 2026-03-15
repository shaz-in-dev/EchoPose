// ============================================================
// ESP32-S3 CSI Node — main.c
// Distributed RF-Inference Mesh Firmware
// ============================================================
// This firmware runs on each ESP32-S3 node.
// It connects to a dedicated Wi-Fi AP, enables CSI on every
// received packet, and streams a compact binary frame over UDP
// to the host aggregator at 20 Hz.
//
// Binary frame layout (per packet, little-endian):
//   [0..3]   magic       uint32  0xCSI1
//   [4..5]   node_id     uint16
//   [6..13]  timestamp   uint64  microseconds since boot
//   [14..15] num_sub     uint16  number of subcarriers (64)
//   [16..N]  csi_data    int16[num_sub*2]  (I,Q) pairs
// ============================================================

#include <string.h>
#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "nvs_flash.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "lwip/sockets.h"
#include "lwip/netdb.h"

#include "csi_capture.h"
#include "udp_sender.h"
#include "ota_updater.h"

static const char *TAG = "main";

// ---- Configuration (override via menuconfig / sdkconfig) ----
#define WIFI_SSID       CONFIG_WIFI_SSID
#define WIFI_PASSWORD   CONFIG_WIFI_PASSWORD
#define NODE_ID         CONFIG_NODE_ID       // 0, 1, or 2
#define HOST_IP         CONFIG_HOST_IP       // e.g. "192.168.1.100"
#define HOST_PORT       CONFIG_HOST_PORT     // e.g. 5005
// -------------------------------------------------------------

static QueueHandle_t csi_queue;

// Wi-Fi event handler
static void wifi_event_handler(void *arg, esp_event_base_t event_base,
                               int32_t event_id, void *event_data)
{
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        ESP_LOGW(TAG, "Wi-Fi disconnected, reconnecting...");
        esp_wifi_connect();
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *event = (ip_event_got_ip_t *)event_data;
        ESP_LOGI(TAG, "Got IP: " IPSTR, IP2STR(&event->ip_info.ip));
    }
}

static void wifi_init(void)
{
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    ESP_ERROR_CHECK(esp_event_handler_instance_register(WIFI_EVENT, ESP_EVENT_ANY_ID,
                                                         &wifi_event_handler, NULL, NULL));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(IP_EVENT, IP_EVENT_STA_GOT_IP,
                                                         &wifi_event_handler, NULL, NULL));

    wifi_config_t wifi_config = {
        .sta = {
            .ssid     = WIFI_SSID,
            .password = WIFI_PASSWORD,
            .threshold.authmode = WIFI_AUTH_WPA2_PSK,
        },
    };
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    // Enable CSI collection
    wifi_csi_config_t csi_config = {
        .lltf_en           = true,
        .htltf_en          = true,
        .stbc_htltf2_en    = true,
        .ltf_merge_en      = true,
        .channel_filter_en = false,
        .manu_scale        = false,
    };
    ESP_ERROR_CHECK(esp_wifi_set_csi_config(&csi_config));
    ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(csi_capture_callback, (void *)csi_queue));
    ESP_ERROR_CHECK(esp_wifi_set_csi(true));

    ESP_LOGI(TAG, "Wi-Fi initialized. SSID: %s, Node ID: %d", WIFI_SSID, NODE_ID);
}

// ---- OTA Polling Task ----
static void ota_polling_task(void *pvParameters) {
    // Wait 30 seconds after boot before checking for the first time
    vTaskDelay(30000 / portTICK_PERIOD_MS);
    
    while(1) {
        char url[128];
        // Poll the HTTP server aggregator for firmware updates
        snprintf(url, sizeof(url), "http://%s:3000/firmware.bin", HOST_IP);
        
        ESP_LOGI(TAG, "Polling for firmware updates at %s", url);
        start_ota_update(url);
        
        // Wait 5 minutes before polling again
        vTaskDelay(300000 / portTICK_PERIOD_MS);
    }
}

void app_main(void)
{
    // Initialize NVS
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    // Create inter-task CSI queue (holds up to 64 frames)
    csi_queue = xQueueCreate(64, sizeof(csi_frame_t));
    configASSERT(csi_queue != NULL);

    // Initialize Wi-Fi and CSI capture
    wifi_init();

    // Start UDP sender task — pinned to core 1, priority 5
    xTaskCreatePinnedToCore(udp_sender_task, "udp_sender",
                            8192, (void *)csi_queue,
                            5, NULL, 1);

    // Start OTA polling task — pinned to core 0, background priority 2
    xTaskCreatePinnedToCore(ota_polling_task, "ota_updater",
                            8192, NULL,
                            2, NULL, 0);

    ESP_LOGI(TAG, "RF-Mesh node %d started. Streaming to %s:%d", NODE_ID, HOST_IP, HOST_PORT);
}
