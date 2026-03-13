#ifndef OTA_UPDATER_H
#define OTA_UPDATER_H

#ifdef __cplusplus
extern "C" {
#endif

// Attempts to download and flash a new firmware bin from the provided HTTP URL.
// If successful, the ESP32 will reboot automatically.
void start_ota_update(const char* url);

#ifdef __cplusplus
}
#endif

#endif // OTA_UPDATER_H
