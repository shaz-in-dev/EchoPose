import struct as s

def make_ico(path):
    sizes = [16, 32, 48, 256]
    bmps  = []
    for sz in sizes:
        w = h = sz
        bpp = 32
        bi = s.pack('<IiiHHIIiiII', 40, w, h * 2, 1, bpp, 0, w * h * 4, 0, 0, 0, 0)
        pixels = []
        for row in range(h - 1, -1, -1):
            for col in range(w):
                # Dark navy background
                r, g, b = 0x0d, 0x15, 0x1e
                # Accent circle in centre
                cx, cy = w / 2, h / 2
                dist = ((col - cx) ** 2 + (row - cy) ** 2) ** 0.5
                ring_outer = w * 0.42
                ring_inner = w * 0.28
                if ring_inner <= dist <= ring_outer:
                    r, g, b = 0x38, 0xbd, 0xf8   # accent blue
                # EP letters (very rough bitmap in larger sizes)
                if sz >= 32:
                    # E: left third
                    lx = int(w * 0.22); rx = int(w * 0.44)
                    ty = int(h * 0.30); by = int(h * 0.70)
                    mid = (ty + by) // 2
                    bar_h = max(2, sz // 16)
                    if lx <= col <= rx:
                        if abs(row - ty) < bar_h or abs(row - mid) < bar_h or abs(row - by) < bar_h:
                            r, g, b = 0xff, 0xff, 0xff
                        if col == lx and ty <= row <= by:
                            r, g, b = 0xff, 0xff, 0xff
                pixels += [b, g, r, 0xff]
        pixel_data = bytes(pixels)
        row_bytes = ((w + 31) // 32) * 4
        and_mask  = bytes(row_bytes * h)
        bmps.append(bi + pixel_data + and_mask)

    count = len(sizes)
    header = s.pack('<HHH', 0, 1, count)
    offset = 6 + count * 16
    entries = b''
    for i, sz in enumerate(sizes):
        data = bmps[i]
        wf = hf = sz if sz < 256 else 0
        entries += s.pack('<BBBBHHII', wf, hf, 0, 0, 1, 32, len(data), offset)
        offset += len(data)
    with open(path, 'wb') as f:
        f.write(header + entries)
        for bmp in bmps:
            f.write(bmp)
    print(f'icon.ico written to {path}')

make_ico(r'c:\Users\Admin\wifi vision\desktop\www\icon.ico')
