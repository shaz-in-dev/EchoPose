# Firmware — ESP32-S3 CSI Node

Each node is an **ESP32-S3** (U.FL antenna version) running ESP-IDF v5.x.

## Prerequisites
- [ESP-IDF v5.x](https://docs.espressif.com/projects/esp-idf/en/latest/esp32s3/get-started/)
- USB-C cable + flashed ESP32-S3 board

## Quick Start

```bash
# 1. Set up ESP-IDF environment
. $IDF_PATH/export.sh

# 2. Configure node (set NODE_ID=0,1,2 and HOST_IP)
idf.py menuconfig

# 3. Build + Flash
idf.py build flash monitor

# Flash node 2 on different port:
NODE_ID=2 idf.py -p COM4 flash
```

## Configuration

Edit `sdkconfig.defaults` or use `idf.py menuconfig`:

| Key | Default | Description |
|-----|---------|-------------|
| `CONFIG_WIFI_SSID` | `rf-mesh-ap` | Dedicated mesh AP SSID |
| `CONFIG_WIFI_PASSWORD` | `rfmesh2025` | AP password |
| `CONFIG_NODE_ID` | `0` | This node's index (0, 1, 2) |
| `CONFIG_HOST_IP` | `192.168.1.100` | Aggregator host IP |
| `CONFIG_HOST_PORT` | `5005` | Aggregator UDP port |

## Frame Format (binary, little-endian)

```
Bytes  0- 3  magic           uint32  0x43534931 ("CSI1")
Bytes  4- 5  node_id         uint16
Bytes  6-13  timestamp       uint64  µs since boot
Bytes 14-15  num_subcarriers uint16  (always 64)
Bytes 16-N   iq_data         int16[] interleaved I,Q pairs
```
