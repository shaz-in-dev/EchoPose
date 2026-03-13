idf_component_register(
    SRCS "main.c" "csi_capture.c" "udp_sender.c" "ota_updater.c"
    INCLUDE_DIRS "."
    REQUIRES esp_wifi nvs_flash lwip esp_timer esp_https_ota esp_http_client app_update
)
