#include "ota_updater.h"
#include "esp_log.h"
#include "esp_system.h"
#include "esp_http_client.h"
#include "esp_https_ota.h"

static const char *TAG = "OTA_UPDATER";

void start_ota_update(const char* url) {
    ESP_LOGI(TAG, "Starting OTA update from %s", url);

    esp_http_client_config_t config = {
        .url = url,
        .cert_pem = NULL, // Skipping cert validation for this local HTTP workflow
        .skip_cert_common_name_check = true,
        .crt_bundle_attach = NULL,
        .timeout_ms = 10000,
        .keep_alive_enable = true,
    };

    esp_https_ota_config_t ota_config = {
        .http_config = &config,
    };

    // Note: We are using esp_https_ota even for HTTP urls as it handles the partition logic cleanly.
    // For a real production system, you MUST use HTTPS with a valid cert_pem.
    esp_err_t ret = esp_https_ota(&ota_config);
    if (ret == ESP_OK) {
        ESP_LOGI(TAG, "OTA update successful! Rebooting...");
        esp_restart();
    } else {
        ESP_LOGE(TAG, "OTA update failed! Error: %s", esp_err_to_name(ret));
    }
}
