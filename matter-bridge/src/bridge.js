/**
 * EchoPose Matter Bridge
 *
 * Exposes EchoPose WiFi CSI analytics as a Matter bridge device so that
 * Apple Home, Google Home, and Amazon Alexa can consume presence, vitals,
 * and fall-detection data without any camera hardware.
 *
 * Matter devices exposed (via AggregatorEndpoint bridge):
 *   - Occupancy Sensor   → presence detection
 *   - Contact Sensor     → fall detected (open = fall)
 *   - Temperature Sensor → heart rate (encoded as × 100 for Matter int16 units)
 *   - Humidity Sensor    → respiratory rate (encoded as × 100)
 *
 * HTTP control API (localhost only, port MATTER_HTTP_PORT):
 *   POST /state    { presence, person_count, heart_rate, rr, fall_detected, activity, stress }
 *   GET  /pairing  → { qrPairingCode, manualPairingCode, qrCodeDataUrl, commissioned }
 *   GET  /status   → { running, commissioned, devices[] }
 *   GET  /health   → 200 OK
 *
 * Environment variables:
 *   MATTER_HTTP_PORT    HTTP API port (default: 7788)
 *   MATTER_PORT         Matter UDP port (default: 5540)
 *   MATTER_PASSCODE     Setup passcode  (default: 20202021)
 *   MATTER_DISCRIMINATOR 12-bit discriminator (default: 3840)
 *   MATTER_STORAGE_DIR  Persistence directory (default: ./matter-storage)
 */

import "@matter/nodejs";

import {
  ServerNode,
  Endpoint,
  VendorId,
  DeviceTypeId,
} from "@matter/main";
import { AggregatorEndpoint } from "@matter/main/endpoints/aggregator";
import { OccupancySensorDevice } from "@matter/main/devices/occupancy-sensor";
import { ContactSensorDevice } from "@matter/main/devices/contact-sensor";
import { TemperatureSensorDevice } from "@matter/main/devices/temperature-sensor";
import { HumiditySensorDevice } from "@matter/main/devices/humidity-sensor";
import { BridgedDeviceBasicInformationServer } from "@matter/main/behaviors/bridged-device-basic-information";
import express from "express";
import QRCode from "qrcode";
import { readFileSync, writeFileSync, existsSync, mkdirSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT_DIR  = join(__dirname, "..");

// ── Config ────────────────────────────────────────────────────────────────────
const HTTP_PORT    = parseInt(process.env.MATTER_HTTP_PORT    ?? "7788");
const MATTER_PORT  = parseInt(process.env.MATTER_PORT         ?? "5540");
const PASSCODE     = parseInt(process.env.MATTER_PASSCODE     ?? "20202021");
const DISCRIMINATOR= parseInt(process.env.MATTER_DISCRIMINATOR?? "3840");
const STORAGE_DIR  = process.env.MATTER_STORAGE_DIR ?? join(ROOT_DIR, "matter-storage");

mkdirSync(STORAGE_DIR, { recursive: true });

// ── Shared state ──────────────────────────────────────────────────────────────
/** Latest analytics pushed by EchoPose inference server */
let latestState = {
  presence:         false,
  person_count:     0,
  heart_rate:       null,   // bpm, or null
  rr:               null,   // breaths/min, or null
  fall_detected:    false,
  activity:         "unknown",
  stress_score:     null,   // 0–100, or null
};

let serverNode      = null;
let commissioned    = false;
let pairingCodes    = null;

// Endpoint refs — populated after bridge creation
const ep = {
  presence:    null,
  fall:        null,
  heartRate:   null,
  respiration: null,
};


// ── Matter bridge setup ───────────────────────────────────────────────────────

async function buildBridge() {
  const aggregator = new Endpoint(AggregatorEndpoint, { id: "echopose-agg" });

  // 1. Occupancy Sensor — presence
  ep.presence = new Endpoint(
    OccupancySensorDevice.with(BridgedDeviceBasicInformationServer),
    {
      id: "presence",
      bridgedDeviceBasicInformation: {
        nodeLabel:  "EchoPose Presence",
        reachable:  true,
        uniqueId:   "ep-presence-001",
        vendorId:   VendorId(0xFFF1),
        productId:  0x8001,
      },
      occupancySensing: {
        occupancy:               { occupied: false },
        occupancySensorType:     0,   // PIR
        occupancySensorTypeBitmap: { pir: true },
      },
    }
  );
  await aggregator.add(ep.presence);

  // 2. Contact Sensor — fall detected
  //    stateValue: false = closed = normal | true = open = FALL ALERT
  ep.fall = new Endpoint(
    ContactSensorDevice.with(BridgedDeviceBasicInformationServer),
    {
      id: "fall",
      bridgedDeviceBasicInformation: {
        nodeLabel: "EchoPose Fall Detector",
        reachable: true,
        uniqueId:  "ep-fall-001",
        vendorId:  VendorId(0xFFF1),
        productId: 0x8002,
      },
      booleanState: {
        stateValue: false,
      },
    }
  );
  await aggregator.add(ep.fall);

  // 3. Temperature Sensor — heart rate
  //    Matter measuredValue = int16 in units of 0.01 °C.
  //    We encode HR (bpm) as HR × 100 so 75 bpm → 7500 (shown as "75.00 °C" in apps).
  //    null → cluster nullValue → controllers show "unavailable".
  ep.heartRate = new Endpoint(
    TemperatureSensorDevice.with(BridgedDeviceBasicInformationServer),
    {
      id: "heart-rate",
      bridgedDeviceBasicInformation: {
        nodeLabel: "EchoPose Heart Rate",
        reachable: true,
        uniqueId:  "ep-hr-001",
        vendorId:  VendorId(0xFFF1),
        productId: 0x8003,
      },
      temperatureMeasurement: {
        measuredValue:    null,
        minMeasuredValue: 3000,    // 30 bpm
        maxMeasuredValue: 25000,   // 250 bpm
      },
    }
  );
  await aggregator.add(ep.heartRate);

  // 4. Humidity Sensor — respiratory rate
  //    Matter measuredValue = uint16 in units of 0.01 %.
  //    We encode RR (brpm) as RR × 100 so 15 brpm → 1500 (shown as "15.00 %").
  ep.respiration = new Endpoint(
    HumiditySensorDevice.with(BridgedDeviceBasicInformationServer),
    {
      id: "respiration",
      bridgedDeviceBasicInformation: {
        nodeLabel: "EchoPose Respiration",
        reachable: true,
        uniqueId:  "ep-rr-001",
        vendorId:  VendorId(0xFFF1),
        productId: 0x8004,
      },
      relativeHumidityMeasurement: {
        measuredValue:    null,
        minMeasuredValue: 0,
        maxMeasuredValue: 10000,   // 100 brpm max
      },
    }
  );
  await aggregator.add(ep.respiration);

  return aggregator;
}


// ── Attribute update helper ───────────────────────────────────────────────────

async function applyState(state) {
  if (!serverNode) return;

  try {
    await ep.presence.set({
      occupancySensing: {
        occupancy: { occupied: state.presence },
      },
    });
  } catch (e) { /* endpoint not yet ready */ }

  try {
    await ep.fall.set({
      booleanState: { stateValue: state.fall_detected },
    });
  } catch (e) {}

  try {
    await ep.heartRate.set({
      temperatureMeasurement: {
        measuredValue: state.heart_rate !== null
          ? Math.round(state.heart_rate * 100)
          : null,
      },
    });
  } catch (e) {}

  try {
    await ep.respiration.set({
      relativeHumidityMeasurement: {
        measuredValue: state.rr !== null
          ? Math.round(state.rr * 100)
          : null,
      },
    });
  } catch (e) {}
}


// ── HTTP control API ──────────────────────────────────────────────────────────

function startHttpApi() {
  const app = express();
  app.use(express.json({ limit: "64kb" }));

  // Bind only to localhost — not exposed externally
  const server = app.listen(HTTP_PORT, "127.0.0.1", () => {
    console.log(`[Matter] HTTP API → http://127.0.0.1:${HTTP_PORT}`);
  });

  /** POST /state — receive EchoPose analytics, update Matter attributes */
  app.post("/state", async (req, res) => {
    const b = req.body;
    latestState = {
      presence:      !!b.presence,
      person_count:  b.person_count  ?? 0,
      heart_rate:    b.heart_rate    ?? null,
      rr:            b.rr            ?? null,
      fall_detected: !!b.fall_detected,
      activity:      b.activity      ?? "unknown",
      stress_score:  b.stress_score  ?? null,
    };
    await applyState(latestState);
    res.json({ ok: true, state: latestState });
  });

  /** GET /pairing — commissioning QR code and manual code */
  app.get("/pairing", async (req, res) => {
    // Already commissioned: QR code is no longer meaningful (and may be null).
    // Return commissioned status clearly so the UI can show the right message.
    if (commissioned) {
      return res.json({
        qrPairingCode:     null,
        manualPairingCode: null,
        qrCodeDataUrl:     null,
        commissioned:      true,
        message:           "EchoPose is already paired with your smart home platform. To re-pair, remove it from your Apple Home / Google Home / Alexa app first.",
      });
    }

    // Not yet commissioned — return pairing codes (available after start())
    if (!pairingCodes) {
      return res.status(503).json({ error: "Bridge initialising — try again in a few seconds" });
    }

    const qrData = pairingCodes.qrPairingCode     ?? "";
    const manual = pairingCodes.manualPairingCode  ?? "";
    let qrCodeDataUrl = null;
    if (qrData) {
      try { qrCodeDataUrl = await QRCode.toDataURL(qrData); } catch (_) {}
    }
    res.json({
      qrPairingCode:     qrData,
      manualPairingCode: manual,
      qrCodeDataUrl,
      commissioned:      false,
    });
  });

  /** GET /status */
  app.get("/status", (req, res) => {
    res.json({
      running:      serverNode !== null,
      commissioned,
      matterPort:   MATTER_PORT,
      devices:      Object.keys(ep),
      latestState,
    });
  });

  /** GET /health — simple liveness probe */
  app.get("/health", (_req, res) => res.json({ ok: true }));

  return server;
}


// ── Main ──────────────────────────────────────────────────────────────────────

async function main() {
  console.log("[Matter] Starting EchoPose Matter Bridge …");

  // Build the aggregator + child sensor endpoints
  const aggregator = await buildBridge();

  // Create the Matter bridge server
  serverNode = await ServerNode.create({
    id: "echopose-bridge",

    network: {
      port: MATTER_PORT,
    },

    commissioning: {
      passcode:      PASSCODE,
      discriminator: DISCRIMINATOR,
    },

    productDescription: {
      name:       "EchoPose Bridge",
      deviceType: DeviceTypeId(0x14),   // Matter Bridge device type
    },

    basicInformation: {
      vendorName:            "EchoPose",
      vendorId:              VendorId(0xFFF1),
      nodeLabel:             "EchoPose WiFi Sensor Hub",
      productName:           "EchoPose",
      productId:             0x8000,
      serialNumber:          "echopose-bridge-001",
      uniqueId:              "echopose-bridge-001",
      hardwareVersion:       1,
      softwareVersion:       1,
      softwareVersionString: "0.2.0",
    },
  });

  // Add the aggregator (all child devices live inside it)
  await serverNode.add(aggregator);

  // Lifecycle hooks
  serverNode.lifecycle.commissioned.on(() => {
    commissioned = true;
    console.log("[Matter] ✓ Device commissioned — it will now appear in Apple Home / Google Home / Alexa");
  });

  serverNode.lifecycle.decommissioned.on(() => {
    commissioned = false;
    console.log("[Matter] Device decommissioned");
  });

  // Start HTTP API before Matter so the Python side can poll /health
  startHttpApi();

  // Start Matter (blocks until process exits)
  await serverNode.start();

  // Pairing codes are available after start()
  try {
    pairingCodes = serverNode.state.commissioning.pairingCodes;
    console.log("\n╔══════════════════════════════════════════╗");
    console.log("║     EchoPose Matter Bridge — READY       ║");
    console.log("╠══════════════════════════════════════════╣");
    console.log(`║  QR code:     ${(pairingCodes?.qrPairingCode ?? "n/a").padEnd(26)} ║`);
    console.log(`║  Manual code: ${(pairingCodes?.manualPairingCode ?? "n/a").padEnd(26)} ║`);
    console.log(`║  Pairing UI:  http://127.0.0.1:${HTTP_PORT}/pairing    ║`);
    console.log("╚══════════════════════════════════════════╝\n");
  } catch (e) {
    console.log("[Matter] Note: could not read pairing codes:", e.message);
  }
}

main().catch((err) => {
  console.error("[Matter] Fatal error:", err);
  process.exit(1);
});
