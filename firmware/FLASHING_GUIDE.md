# EchoPose V3 Flashing Guide (Waveshare ESP32-S3 Nano)

Follow these steps to flash your two ESP32-S3 Nano boards and begin live testing.

## 1. Environment Setup
Ensure you have the **ESP-IDF** (v5.x recommended) installed on your system.
- [ESP-IDF Get Started Guide](https://docs.espressif.com/projects/esp-idf/en/stable/esp32s3/get-started/index.html)

Once installed, open your terminal (IDF PowerShell or CMD) and navigate to the `firmware` directory:
```bash
cd "c:\Users\Admin\wifi vision\firmware"
```

## 2. Global Configuration
The firmware pulls constants from `sdkconfig`. You can set your Wi-Fi credentials and Host IP globally first:

1. Run the configuration menu:
   ```bash
   idf.py menuconfig
   ```
2. Navigate to **"EchoPose Configuration"** (if defined) or **"Component Config"** -> **"RF-Mesh Node Config"**.
3. Set the following:
   - **SSID:** Your 2.4GHz Wi-Fi Name.
   - **Password:** Your Wi-Fi Password.
   - **Host IP:** The IP address of the machine running the Rust Aggregator.
   - **Host Port:** `5005` (Default).

> [!IMPORTANT]
> Ensure your machine and the ESP32s are on the same network and the firewall allows UDP on port 5005.

---

## 3. Flashing Node 0 (Transmitter/Receiver)
Connect your first ESP32-S3 Nano.

1. Set the Node ID to `0` in `menuconfig`.
2. Build and Flash:
   ```bash
   idf.py build
   idf.py -p COM_PORT flash monitor
   ```
   *(Replace `COM_PORT` with your port, e.g., `COM3` on Windows)*

## 4. Flashing Node 1 (Receiver)
Connect your second ESP32-S3 Nano.

1. Re-open `menuconfig` and change **Node ID** to `1`.
2. Save and Flash:
   ```bash
   idf.py -p COM_PORT flash monitor
   ```

---

## 5. Testing the Stream
Once flashed, both nodes will boot and attempt to connect to Wi-Fi. In the `monitor`, you should see:
```text
I (main) Got IP: 192.168.1.XX
I (main) RF-Mesh node 0 started. Streaming to 192.168.1.YY:5005
```

### Server Side Checklist
1. **Start Aggregator:**
   ```bash
   cd aggregator && cargo run --release
   ```
2. **Start Inference:**
   ```bash
   cd inference && python server_v2.py
   ```
3. **Open UI:**
   Open `ui/index.html` in your browser.

## 6. Physical Placement
Place **Node 0** and **Node 1** on opposite sides of the room (e.g., 2-3 meters apart) at waist height. The system will detect a human walking between them.
