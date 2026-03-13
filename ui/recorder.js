// ui/recorder.js
let isRecording = false;
let recordedFrames = [];
let recordStartTime = 0;

const btnRecord = document.getElementById('btn-record');
const btnStop = document.getElementById('btn-stop');
const filePlayback = document.getElementById('file-playback');

if (btnRecord && btnStop) {
  btnRecord.addEventListener('click', () => {
    isRecording = true;
    recordedFrames = [];
    recordStartTime = performance.now();
    btnRecord.classList.add('hidden');
    btnStop.classList.remove('hidden');
    console.log("Recording started...");
  });

  btnStop.addEventListener('click', () => {
    isRecording = false;
    btnRecord.classList.remove('hidden');
    btnStop.classList.add('hidden');
    
    if (recordedFrames.length === 0) return;
    
    const blob = new Blob([JSON.stringify(recordedFrames)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `echopose_recording_${new Date().toISOString().replace(/[:.]/g, '-')}.json`;
    a.click();
    URL.revokeObjectURL(url);
    console.log("Recording saved. Frames:", recordedFrames.length);
  });
}

function recordFrame(data) {
  if (!isRecording) return;
  // Deep copy the data and add elapsed time
  const frameCopy = JSON.parse(JSON.stringify(data));
  frameCopy.rel_time = performance.now() - recordStartTime;
  recordedFrames.push(frameCopy);
}

// Playback logic
if (filePlayback) {
  filePlayback.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = (evt) => {
      try {
        const frames = JSON.parse(evt.target.result);
        if (frames.length > 0) {
          playRecording(frames);
        }
      } catch (err) {
        console.error("Failed to parse recording", err);
      }
    };
    reader.readAsText(file);
    e.target.value = ""; // reset
  });
}

let playbackTimer = null;
function playRecording(frames) {
  // If we are currently connected, disconnect to avoid interference
  if (window.ws) {
    window.intendedDisconnect = true;
    window.ws.close();
  }
  
  clearTimeout(playbackTimer);
  const statusBadge = document.getElementById('status-badge');
  if (statusBadge) {
      statusBadge.textContent = 'Playback';
      statusBadge.className = 'badge badge--connected';
  }
  
  let currentIndex = 0;
  const startTime = performance.now();
  
  function loop() {
    if (currentIndex >= frames.length) {
      if (statusBadge) {
        statusBadge.textContent = 'Playback Ended';
        statusBadge.className = 'badge badge--disconnected';
      }
      return;
    }
    
    const now = performance.now();
    const elapsed = now - startTime;
    const nextFrame = frames[currentIndex];
    
    if (elapsed >= nextFrame.rel_time) {
      if (typeof window.handleFrame === 'function') {
        window.handleFrame(nextFrame);
      }
      currentIndex++;
      while (currentIndex < frames.length && elapsed >= frames[currentIndex].rel_time) {
        if (typeof window.handleFrame === 'function') {
          window.handleFrame(frames[currentIndex]);
        }
        currentIndex++;
      }
    }
    
    playbackTimer = requestAnimationFrame(loop);
  }
  
  playbackTimer = requestAnimationFrame(loop);
}
