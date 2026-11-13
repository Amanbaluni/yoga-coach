import { PoseLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";
import { T, ANGLE_NAMES, frameFeatures, angles, canon } from "./features.js";

const WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const TASK = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task";
const $ = id => document.getElementById(id);
const video = $("video"), canvas = $("overlay"), ctx = canvas.getContext("2d");
const BONES = [[11,12],[11,13],[13,15],[12,14],[14,16],[11,23],[12,24],[23,24],[23,25],[25,27],[24,26],[26,28]];
const NEED = [11,12,13,14,15,16,23,24,25,26,27,28];
let model, labels, ideal, landmarker, target, lastT = -1, buf = [], recent = [], holdStart = null, lastSpoken = 0, running = false, stream = null;

// Stick-figure glyphs (48x48) so poses can be picked by shape. Unknown poses fall back to a standing figure.
const GLYPH = {
  tree:    '<circle class="hd" cx="24" cy="9" r="2.6"/><path d="M15 3l9 14 9-14M24 17v12v16M24 29l11 6-10 4"/>',
  warrior: '<circle class="hd" cx="24" cy="9" r="2.8"/><path d="M5 17h38M24 14v14M24 28l12 3v13M24 28 10 44"/>',
  goddess: '<circle class="hd" cx="24" cy="8" r="2.8"/><path d="M24 13v14M24 27l13 8v9M24 27 11 35v9M24 16h9V8M24 16h-9V8"/>',
  plank:   '<circle class="hd" cx="7" cy="22" r="3"/><path d="M12 25 27 29l16 5M12 25v15M43 34v6"/>',
  downdog: '<circle class="hd" cx="21" cy="29" r="2.8"/><path d="M8 41 17 22 28 8l16 33"/>',
};
const GLYPH_FALLBACK = '<circle class="hd" cx="24" cy="8" r="3"/><path d="M24 14v15M24 29l-6 15M24 29l6 15M12 20h24"/>';
const HINT = {
  tree: "Stand on one leg, foot against the inner thigh, hands together overhead.",
  warrior: "Feet wide, front knee bent over the ankle, arms level with the floor.",
  goddess: "Feet wide and turned out, knees bent over your toes, elbows bent at shoulder height.",
  plank: "Hands under shoulders, body in one straight line from head to heels.",
  downdog: "Hips high, arms and legs long, forming an upside-down V.",
};
const pretty = s => s.replace(/[_-]+/g, " ").replace(/^./, c => c.toUpperCase());
const store = (k, v) => { try { if (v === undefined) return +localStorage.getItem(k) || 0; localStorage.setItem(k, v); } catch { return 0; } };
const status = (t, m = "") => { $("status").textContent = t; $("status").dataset.mode = m; };
let lastMsg = "";
function speak(t) {                           // spoken tips: only when the toggle is on, never repeat the same line within 6s
  if (!$("voice").checked || !("speechSynthesis" in window) || !t) return;
  const now = performance.now();
  if (t === lastMsg && now - lastSpoken < 6000) return;
  if (t !== lastMsg && now - lastSpoken < 2500) return;
  lastMsg = t; lastSpoken = now;
  say(t);
}
function say(t, test) {
  const u = new SpeechSynthesisUtterance(t), v = speechSynthesis.getVoices();
  const voice = v.find(x => /^en/i.test(x.lang) && x.default) || v.find(x => /^en/i.test(x.lang)) || v[0];
  if (voice) { u.voice = voice; u.lang = voice.lang; } else u.lang = "en-US";
  u.onerror = e => { if (e.error !== "canceled" && e.error !== "interrupted") status("Voice failed (" + e.error + "). Your browser may have no speech voices installed.", "fix"); };
  if (test) { u.onstart = () => status("Spoken tips on."); if (!v.length) status("No speech voices found on this device/browser, so spoken tips cannot play. Try Chrome or Edge, or install a text-to-speech voice in your system settings.", "fix"); }
  speechSynthesis.cancel(); speechSynthesis.resume(); speechSynthesis.speak(u);
}
const setBest = (l, v) => { if (l === target) $("best").textContent = v.toFixed(1); };

async function loadAI() {
  try {
    const j = f => fetch(f).then(r => { if (!r.ok) throw new Error(f); return r.json(); });
    [labels, ideal] = await Promise.all([j("labels.json"), j("ideal.json")]);
    model = await tf.loadLayersModel("model/model.json");
    model.predict(tf.zeros([1, T, model.inputs[0].shape[2]])).dispose();   // warm-up
    labels.forEach(l => $("poses").append(new Option(pretty(l), l)));
    $("poses").onchange = e => select(e.target.value);
    select(labels[0]); $("start").disabled = false; $("start").textContent = "Start camera"; status("AI model ready. Start the camera.");
  } catch (e) {
    status("Trained model not found (" + e.message + "). Train it first, see README."); $("start").textContent = "Model missing";
  }
}
function select(l) {
  target = l; holdStart = null;
  $("poses").value = l; $("poseGlyph").innerHTML = GLYPH[l] || GLYPH_FALLBACK;
  $("tip").textContent = HINT[l] || "Hold " + pretty(l) + " and the coach will score how closely you match it.";
  $("pct").textContent = "0"; $("ring").style.setProperty("--p", 0); $("ring").setAttribute("aria-valuenow", 0);
  if (landmarker) $("detected").textContent = "Waiting for you";
  $("best").textContent = store("best-" + l).toFixed(1); $("hold").textContent = "0.0"; rows([]);
}
function rows(list, ok) {
  $("checks").innerHTML = "";
  list.forEach(t => { const li = document.createElement("li"); li.className = ok ? "ok" : "bad"; li.textContent = t; $("checks").append(li); });
}
function corrections(a) {                     // data-driven: compare with the ideal angles learned from the dataset
  const sw = a[0] > a[1], id = ideal[target], out = [];
  canon(a).forEach((v, i) => {
    const d = v - id.median[i], tol = Math.max(1.5 * id.std[i], 15);
    if (Math.abs(d) <= tol) return;
    const name = ANGLE_NAMES[sw ? i ^ 1 : i].toLowerCase();
    const msg = i === 4 || i === 5 ? `${d > 0 ? "Lower" : "Raise"} your ${name.split(" ")[0]} arm`
      : `${d > 0 ? "Bend" : "Straighten"} your ${name} more`;
    out.push([Math.abs(d) / tol, msg]);
  });
  return out.sort((x, y) => y[0] - x[0]).slice(0, 3).map(x => x[1]);
}
function draw(p, color) {
  const k = canvas.width / 640;                // keep line weight the same on screen whatever the camera resolution
  ctx.lineCap = "round";
  [["rgba(8,14,12,.55)", 9], [color, 5]].forEach(([c, w]) => {   // dark under-stroke keeps the skeleton readable on any background
    ctx.lineWidth = w * k; ctx.strokeStyle = c;
    BONES.forEach(([i, j]) => { ctx.beginPath(); ctx.moveTo(p[i].x, p[i].y); ctx.lineTo(p[j].x, p[j].y); ctx.stroke(); });
  });
  ctx.fillStyle = "#fff"; NEED.forEach(i => { ctx.beginPath(); ctx.arc(p[i].x, p[i].y, 5.5 * k, 0, 7); ctx.fill(); });
}
function onFrame(lm) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!lm) { status("No person found. Step into the frame."); buf = []; recent = []; holdStart = null; return; }
  const p = lm.map(l => ({ x: l.x * canvas.width, y: l.y * canvas.height, v: l.visibility }));
  // Lenient gate: the training photos had no visibility check, and in Down Dog the far-side limbs are
  // often occluded (low visibility) even though BlazePose still places them correctly.
  const hidden = NEED.filter(i => p[i].v < 0.25 || p[i].x < -20 || p[i].x > canvas.width + 20 || p[i].y < -20 || p[i].y > canvas.height + 20).length;
  if (hidden > 3) { draw(p, "#b9c4bf"); status("Step back until your whole body is visible."); speak("Step back until your whole body is visible"); buf = []; recent = []; holdStart = null; return; }
  buf.push(frameFeatures(p)); if (buf.length > T) buf.shift();
  if (buf.length < T) { draw(p, "#b9c4bf"); status("Analysing…"); return; }

  const raw = tf.tidy(() => Array.from(model.predict(tf.tensor3d([buf])).dataSync()));
  recent.push(raw); if (recent.length > 5) recent.shift();           // smooth over 5 predictions
  const probs = raw.map((_, k) => recent.reduce((s, r) => s + r[k], 0) / recent.length);
  const top = probs.indexOf(Math.max(...probs)), ti = labels.indexOf(target);
  const ok = top === ti && probs[top] > 0.6, pct = probs[ti] * 100;
  $("pct").textContent = Math.round(pct); $("ring").style.setProperty("--p", pct); $("ring").setAttribute("aria-valuenow", Math.round(pct));
  $("ring").style.setProperty("--c", ok ? "var(--good)" : "var(--fix)");
  $("detected").textContent = `Detected ${pretty(labels[top])}, ${Math.round(probs[top] * 100)}%`;
  draw(p, ok ? "#5fd1a8" : "#f0a93f");

  const now = performance.now();
  if (ok) {
    holdStart ??= now; const s = (now - holdStart) / 1000; $("hold").textContent = s.toFixed(1);
    if (s > store("best-" + target)) { store("best-" + target, s); setBest(target, s); }
    rows(["Great form. Hold it."], true); status("Great form. Hold it.", "good"); if (s < 0.5) speak("Great form. Hold it.");
  } else {
    holdStart = null; $("hold").textContent = "0.0";
    const fix = corrections(angles(p)); rows(fix.length ? fix : ["Adjust until the AI recognises the pose"]);
    status(fix[0] || "Keep adjusting", "fix");
    speak(fix[0] || "Keep adjusting");
  }
}
function loop() {
  if (!running) return;
  if (video.currentTime !== lastT) { lastT = video.currentTime; onFrame(landmarker.detectForVideo(video, performance.now()).landmarks[0]); }
  requestAnimationFrame(loop);
}
async function createLandmarker() {
  const fs = await FilesetResolver.forVisionTasks(WASM);
  const o = d => ({ baseOptions: { modelAssetPath: TASK, delegate: d }, runningMode: "VIDEO", numPoses: 1 });
  try { return await PoseLandmarker.createFromOptions(fs, o("GPU")); } catch { return await PoseLandmarker.createFromOptions(fs, o("CPU")); }
}
$("start").onclick = async () => {
  const btn = $("start"); btn.disabled = true; status("Loading pose tracker…");
  try {
    landmarker ??= await createLandmarker();
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } }, audio: false });
    video.srcObject = stream;
    await video.play(); canvas.width = video.videoWidth; canvas.height = video.videoHeight;
    const fr = $("frame").style, ar = video.videoWidth / video.videoHeight;   // match the stage to the real camera shape (portrait phones too)
    fr.setProperty("--ar", ar); fr.setProperty("--ar-n", ar);
    $("intro").hidden = true; $("stop").hidden = false; document.body.dataset.live = "1"; status("Get into position.");
    running = true; lastT = -1; loop();
  } catch (e) { btn.disabled = false; status(e.name === "NotAllowedError" ? "Camera blocked. Allow camera access and try again." : "Could not start: " + e.message); }
};
$("stop").onclick = () => {
  running = false; stream?.getTracks().forEach(t => t.stop()); video.srcObject = null; stream = null;
  ctx.clearRect(0, 0, canvas.width, canvas.height); buf = []; recent = []; holdStart = null; speechSynthesis?.cancel();
  $("hold").textContent = "0.0"; $("pct").textContent = "0"; $("ring").style.setProperty("--p", 0); rows([]);
  $("detected").textContent = "Waiting for you"; $("stop").hidden = true; $("intro").hidden = false; delete document.body.dataset.live;
  $("start").disabled = false; $("start").textContent = "Start camera"; status("Camera off. Start it when you are ready.");
};
$("voice").onchange = () => { lastSpoken = 0; lastMsg = ""; if ($("voice").checked) { say("Spoken tips on", true); } else speechSynthesis.cancel(); };
if ("speechSynthesis" in window) speechSynthesis.getVoices();   // start loading voices early
if (!("speechSynthesis" in window)) { $("voice").disabled = true; $("voice").parentElement.title = "Speech is not supported in this browser"; }
$("start").disabled = true; loadAI();
