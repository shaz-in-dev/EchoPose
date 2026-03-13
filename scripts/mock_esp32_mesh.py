import socket
import struct
import time
import math
import random

# Configuration
HOST = "127.0.0.1"
PORT = 5005
NODES = [0, 1, 2]
RATE_HZ = 20  # Total rate per node
MAGIC = 0x43534931  # "CSI1"
NUM_SUBCARRIERS = 64

def generate_csi_packet(node_id, t):
    """
    Constructs a binary CSI packet compatible with the Rust Aggregator.
    Format:
    - magic:           u32 (4 bytes)
    - node_id:         u16 (2 bytes)
    - timestamp_us:    u64 (8 bytes)
    - num_subcarriers: u16 (2 bytes)
    - iq_data:         [i16; 64 * 2] (256 bytes)
    """
    timestamp_us = int(time.time() * 1_000_000)
    
    # Generate some synthetic wave data
    # We'll create a shifting phase to simulate "movement"
    iq_data = []
    for i in range(NUM_SUBCARRIERS):
        # Base signal + some noise + motion effect
        base_phase = (t * 2.0) + (i * 0.1) + (node_id * 0.5)
        amp = 1000 + 200 * math.sin(t * 1.5 + i * 0.05)
        
        re = int(amp * math.cos(base_phase))
        im = int(amp * math.sin(base_phase))
        
        # Clamp to i16
        re = max(-32768, min(32767, re))
        im = max(-32768, min(32767, im))
        
        iq_data.extend([re, im])
    
    # Pack the header
    # < = little-endian
    # I = u32, H = u16, Q = u64
    header = struct.pack("<I H Q H", MAGIC, node_id, timestamp_us, NUM_SUBCARRIERS)
    
    # Pack the IQ data (64 * 2 = 128 shorts)
    data_payload = struct.pack(f"<{len(iq_data)}h", *iq_data)
    
    return header + data_payload

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"🚀 Starting RF-Mesh ESP32 Simulator")
    print(f"Sending dummy CSI packets to {HOST}:{PORT}")
    print(f"Simulating {len(NODES)} nodes at {RATE_HZ}Hz each...")
    print("Press Ctrl+C to stop.")

    start_time = time.time()
    try:
        while True:
            t = time.time() - start_time
            for node_id in NODES:
                packet = generate_csi_packet(node_id, t)
                sock.sendto(packet, (HOST, PORT))
            
            # Sleep to maintain frequency
            time.sleep(1.0 / RATE_HZ)
            
    except KeyboardInterrupt:
        print("\nStopping simulator...")
    finally:
        sock.close()

if __name__ == "__main__":
    main()
