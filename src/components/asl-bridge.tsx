"use client";

import {
  Activity,
  ArrowDownLeft,
  ArrowDownRight,
  ArrowLeftRight,
  AudioLines,
  BookOpen,
  ChevronDown,
  CircleHelp,
  Clipboard,
  Download,
  ExternalLink,
  FileDown,
  Fingerprint,
  Gauge,
  Hand,
  Keyboard,
  Layers3,
  LibraryBig,
  LockKeyhole,
  MessageSquareText,
  Plus,
  ScanLine,
  Search,
  Settings2,
  ShieldCheck,
  Sparkles,
  Square,
  Video,
  Volume2,
  X,
  Zap,
} from "lucide-react";
import type { HandLandmarker } from "@mediapipe/tasks-vision";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { findSignByPhrase, signLibrary, type SignEntry } from "@/lib/sign-library";

type View = "translate" | "library" | "model";
type Direction = "sign-to-text" | "text-to-sign";
type Prediction = { label: string; confidence: number; status: string } | null;
type Sample = {
  label: string;
  features: number[];
  signer_id: string;
  captured_at: string;
  feature_version: "wrist-relative-63-v1";
};

type HandPoint = { x: number; y: number; z: number };

const STORAGE_KEY = "asl-bridge.samples.v1";
const LANDMARK_CONNECTIONS: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

function normalizeLandmarks(points: HandPoint[]): number[] {
  if (points.length !== 21) return [];
  const wrist = points[0];
  let scale = 0;
  for (const point of points) {
    scale = Math.max(scale, Math.hypot(point.x - wrist.x, point.y - wrist.y));
  }
  if (!Number.isFinite(scale) || scale < 0.0001) return [];
  return points.flatMap((point) => [
    Number(((point.x - wrist.x) / scale).toFixed(6)),
    Number(((point.y - wrist.y) / scale).toFixed(6)),
    Number(((point.z - wrist.z) / scale).toFixed(6)),
  ]);
}

function readSamples(): Sample[] {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter((item): item is Sample => {
      return Boolean(
        item && typeof item === "object" &&
        typeof (item as Sample).label === "string" &&
        Array.isArray((item as Sample).features) &&
        (item as Sample).features.length === 63 &&
        typeof (item as Sample).signer_id === "string",
      );
    });
  } catch {
    return [];
  }
}

function initials(text: string) {
  return text.split(/\s+/).filter(Boolean).slice(0, 2).map((part) => part[0]).join("").toUpperCase();
}

function createPracticeSequence(text: string) {
  const clean = text.trim();
  if (!clean) return { tokens: [] as { text: string; known: boolean; entry?: SignEntry }[], exact: undefined as SignEntry | undefined };
  const exact = findSignByPhrase(clean);
  if (exact) {
    return { exact, tokens: exact.gloss.map((word) => ({ text: word, known: true, entry: exact })) };
  }
  const parts = clean.replace(/[?.!,]/g, " ").split(/\s+/).filter(Boolean);
  const tokens = parts.map((word) => {
    const entry = findSignByPhrase(word);
    return entry ? { text: entry.gloss[0], known: true, entry } : { text: word.toUpperCase(), known: false };
  });
  return { exact: undefined, tokens };
}

export default function AslBridge() {
  const [view, setView] = useState<View>("translate");
  const [direction, setDirection] = useState<Direction>("sign-to-text");
  const [cameraOn, setCameraOn] = useState(false);
  const [cameraLoading, setCameraLoading] = useState(false);
  const [cameraError, setCameraError] = useState("");
  const [handsDetected, setHandsDetected] = useState(0);
  const [detectorReady, setDetectorReady] = useState(false);
  const [sampleLabel, setSampleLabel] = useState("HELLO");
  const [signerId, setSignerId] = useState("signer-01");
  const [samples, setSamples] = useState<Sample[]>([]);
  const [sampleNotice, setSampleNotice] = useState("");
  const [apiEnabled, setApiEnabled] = useState(false);
  const [apiStatus, setApiStatus] = useState<"off" | "checking" | "ready" | "no-model" | "offline">("off");
  const [prediction, setPrediction] = useState<Prediction>(null);
  const [phraseInput, setPhraseInput] = useState("Hello");
  const [practiceBuilt, setPracticeBuilt] = useState(false);
  const [libraryQuery, setLibraryQuery] = useState("");
  const [libraryCategory, setLibraryCategory] = useState("All");
  const [transcript, setTranscript] = useState<string[]>([]);
  const [notice, setNotice] = useState("");

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const landmarkerRef = useRef<HandLandmarker | null>(null);
  const rafRef = useRef<number | null>(null);
  const currentFeaturesRef = useRef<number[] | null>(null);
  const frameRef = useRef(0);
  const apiPendingRef = useRef(false);
  const apiBase = process.env.NEXT_PUBLIC_ASL_API_URL || "http://127.0.0.1:8000";
  const apiEnabledRef = useRef(false);
  const apiStatusRef = useRef<"off" | "checking" | "ready" | "no-model" | "offline">("off");

  useEffect(() => {
    apiEnabledRef.current = apiEnabled;
  }, [apiEnabled]);

  useEffect(() => {
    apiStatusRef.current = apiStatus;
  }, [apiStatus]);

  useEffect(() => {
    setSamples(readSamples());
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
      streamRef.current?.getTracks().forEach((track) => track.stop());
      landmarkerRef.current?.close();
    };
  }, []);

  const stopCamera = useCallback(() => {
    if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;
    landmarkerRef.current?.close();
    landmarkerRef.current = null;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    const canvas = canvasRef.current;
    const context = canvas?.getContext("2d");
    if (canvas && context) context.clearRect(0, 0, canvas.width, canvas.height);
    currentFeaturesRef.current = null;
    setCameraOn(false);
    setCameraLoading(false);
    setDetectorReady(false);
    setHandsDetected(0);
  }, []);

  const askForPrediction = useCallback(async (features: number[]) => {
    if (apiPendingRef.current || !apiEnabledRef.current) return;
    apiPendingRef.current = true;
    try {
      const response = await fetch(`${apiBase.replace(/\/$/, "")}/api/v1/recognize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ features }),
      });
      const body = await response.json() as { status?: string; prediction?: { label?: string; confidence?: number }; detail?: string };
      if (body.status === "model_unavailable") {
        setApiStatus("no-model");
        setPrediction(null);
      } else if (response.ok && body.prediction?.label) {
        setApiStatus("ready");
        setPrediction({ label: body.prediction.label, confidence: body.prediction.confidence ?? 0, status: body.status ?? "recognized" });
      } else {
        setPrediction(null);
      }
    } catch {
      setApiStatus("offline");
      setPrediction(null);
    } finally {
      apiPendingRef.current = false;
    }
  }, [apiBase]);

  const runFrameLoop = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const detector = landmarkerRef.current;
    if (!video || !canvas || !detector) return;
    if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
      rafRef.current = requestAnimationFrame(runFrameLoop);
      return;
    }

    const result = detector.detectForVideo(video, performance.now());
    const context = canvas.getContext("2d");
    const width = video.videoWidth || 640;
    const height = video.videoHeight || 480;
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    if (context) {
      context.clearRect(0, 0, width, height);
      context.lineCap = "round";
      context.lineJoin = "round";
      for (const hand of result.landmarks) {
        context.strokeStyle = "rgba(69, 239, 205, 0.9)";
        context.lineWidth = Math.max(2, width / 400);
        for (const [from, to] of LANDMARK_CONNECTIONS) {
          const a = hand[from];
          const b = hand[to];
          if (!a || !b) continue;
          context.beginPath();
          context.moveTo(a.x * width, a.y * height);
          context.lineTo(b.x * width, b.y * height);
          context.stroke();
        }
        for (let index = 0; index < hand.length; index += 1) {
          const point = hand[index];
          context.beginPath();
          context.fillStyle = [4, 8, 12, 16, 20].includes(index) ? "#e9fff9" : "#36e6c4";
          context.arc(point.x * width, point.y * height, Math.max(3, width / 240), 0, Math.PI * 2);
          context.fill();
        }
      }
    }

    const firstHand = result.landmarks[0] as HandPoint[] | undefined;
    currentFeaturesRef.current = firstHand ? normalizeLandmarks(firstHand) : null;
    frameRef.current += 1;
    if (frameRef.current % 8 === 0) setHandsDetected(result.landmarks.length);
    if (apiEnabledRef.current && apiStatusRef.current === "ready" && currentFeaturesRef.current && frameRef.current % 12 === 0) {
      void askForPrediction(currentFeaturesRef.current);
    }
    rafRef.current = requestAnimationFrame(runFrameLoop);
  }, [askForPrediction]);

  const startCamera = useCallback(async () => {
    setCameraError("");
    setCameraLoading(true);
    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error("Camera access is not available in this browser. Use a current browser on localhost or HTTPS.");
      }
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
      });
      streamRef.current = stream;
      const video = videoRef.current;
      if (!video) throw new Error("Camera preview could not be initialized.");
      video.srcObject = stream;
      await video.play();

      const vision = await import("@mediapipe/tasks-vision");
      const fileset = await vision.FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22/wasm",
      );
      landmarkerRef.current = await vision.HandLandmarker.createFromOptions(fileset, {
        baseOptions: {
          modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
          delegate: "GPU",
        },
        runningMode: "VIDEO",
        numHands: 2,
        minHandDetectionConfidence: 0.55,
        minHandPresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });
      setCameraOn(true);
      setDetectorReady(true);
      setCameraLoading(false);
      frameRef.current = 0;
      rafRef.current = requestAnimationFrame(runFrameLoop);
    } catch (error) {
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
      if (videoRef.current) videoRef.current.srcObject = null;
      setCameraLoading(false);
      setCameraOn(false);
      setCameraError(error instanceof Error ? error.message : "Could not start camera tracking.");
    }
  }, [runFrameLoop]);

  const captureSample = useCallback(() => {
    const features = currentFeaturesRef.current;
    if (!features || features.length !== 63) {
      setSampleNotice("No hand detected yet. Hold one hand inside the frame and try again.");
      return;
    }
    const label = sampleLabel.trim().toUpperCase();
    if (!label) {
      setSampleNotice("Enter a label first, for example HELLO or A.");
      return;
    }
    const sample: Sample = {
      label,
      features: [...features],
      signer_id: signerId.trim() || "unspecified-signer",
      captured_at: new Date().toISOString(),
      feature_version: "wrist-relative-63-v1",
    };
    const updated = [...samples, sample];
    setSamples(updated);
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
      setSampleNotice(`Saved one ${label} landmark sample locally. No video was saved.`);
    } catch {
      setSamples(samples);
      setSampleNotice("Browser storage is full or unavailable. Export or clear old samples, then try again.");
    }
  }, [sampleLabel, samples, signerId]);

  const exportSamples = useCallback(() => {
    if (!samples.length) {
      setSampleNotice("Capture at least one sample before exporting.");
      return;
    }
    const payload = samples.map((sample) => JSON.stringify(sample)).join("\n") + "\n";
    const blob = new Blob([payload], { type: "application/x-ndjson" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "asl-bridge-samples.jsonl";
    anchor.click();
    URL.revokeObjectURL(url);
    setSampleNotice(`Exported ${samples.length} labeled samples as JSONL.`);
  }, [samples]);

  const clearSamples = useCallback(() => {
    if (!window.confirm("Delete all locally saved landmark samples? This cannot be undone.")) return;
    window.localStorage.removeItem(STORAGE_KEY);
    setSamples([]);
    setSampleNotice("Local sample set cleared.");
  }, []);

  const toggleApi = useCallback(async () => {
    if (apiEnabled) {
      setApiEnabled(false);
      setApiStatus("off");
      setPrediction(null);
      return;
    }
    setApiEnabled(true);
    setApiStatus("checking");
    try {
      const response = await fetch(`${apiBase.replace(/\/$/, "")}/api/v1/health`);
      if (!response.ok) throw new Error("API unavailable");
      const body = await response.json() as { model_loaded?: boolean };
      setApiStatus(body.model_loaded ? "ready" : "no-model");
    } catch {
      setApiStatus("offline");
    }
  }, [apiBase, apiEnabled]);

  const addPredictionToTranscript = useCallback(() => {
    if (!prediction?.label) return;
    setTranscript((items) => [...items, prediction.label]);
    setNotice(`Added “${prediction.label}” to the transcript.`);
  }, [prediction]);

  const speakText = useCallback((text: string) => {
    if (!text.trim() || !("speechSynthesis" in window)) {
      setNotice("Speech synthesis is unavailable in this browser.");
      return;
    }
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = "en-US";
    window.speechSynthesis.speak(utterance);
  }, []);

  const builtSequence = useMemo(() => createPracticeSequence(phraseInput), [phraseInput]);
  const labelCounts = useMemo(() => {
    const counts = new Map<string, number>();
    for (const sample of samples) counts.set(sample.label, (counts.get(sample.label) ?? 0) + 1);
    return [...counts.entries()].sort((a, b) => b[1] - a[1]);
  }, [samples]);
  const filteredLibrary = useMemo(() => {
    const query = libraryQuery.trim().toLowerCase();
    return signLibrary.filter((entry) => {
      const categoryMatch = libraryCategory === "All" || entry.category === libraryCategory;
      const queryMatch = !query || `${entry.phrase} ${entry.gloss.join(" ")} ${entry.note}`.toLowerCase().includes(query);
      return categoryMatch && queryMatch;
    });
  }, [libraryCategory, libraryQuery]);

  const openPractice = (entry: SignEntry) => {
    setPhraseInput(entry.phrase);
    setPracticeBuilt(true);
    setDirection("text-to-sign");
    setView("translate");
  };

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <button className="brand-lockup" onClick={() => { setView("translate"); setDirection("sign-to-text"); }} aria-label="ASL Bridge home">
          <span className="brand-mark"><Hand size={21} strokeWidth={2.5} /></span>
          <span className="brand-copy"><strong>asl<span>.</span>bridge</strong><small>COMMUNICATION LAB</small></span>
        </button>

        <div className="workspace-label">WORKSPACE</div>
        <nav className="side-nav" aria-label="Main navigation">
          <button className={`nav-item ${view === "translate" ? "active" : ""}`} onClick={() => setView("translate")}>
            <ScanLine size={18} /><span>Translator</span><kbd>⌘1</kbd>
          </button>
          <button className={`nav-item ${view === "library" ? "active" : ""}`} onClick={() => setView("library")}>
            <LibraryBig size={18} /><span>Sign library</span><span className="nav-count">{signLibrary.length}</span>
          </button>
          <button className={`nav-item ${view === "model" ? "active" : ""}`} onClick={() => setView("model")}>
            <Activity size={18} /><span>Model lab</span><span className="nav-dot" />
          </button>
        </nav>

        <div className="sidebar-divider" />
        <div className="workspace-label">QUICK START</div>
        <button className="quick-card" onClick={() => { setView("translate"); setDirection("sign-to-text"); }}>
          <span className="quick-icon mint"><Video size={16} /></span>
          <span><strong>Live camera</strong><small>Track hand landmarks</small></span>
          <ArrowDownRight size={15} />
        </button>
        <button className="quick-card" onClick={() => { setView("translate"); setDirection("text-to-sign"); }}>
          <span className="quick-icon violet"><Keyboard size={16} /></span>
          <span><strong>Text to gloss</strong><small>Build a practice queue</small></span>
          <ArrowDownRight size={15} />
        </button>

        <div className="sidebar-spacer" />
        <div className="privacy-card">
          <div className="privacy-icon"><ShieldCheck size={18} /></div>
          <strong>Privacy by default</strong>
          <p>Camera frames stay in your browser. Landmark export is opt-in.</p>
          <span className="privacy-status"><span /> LOCAL-FIRST WORKSPACE</span>
        </div>
        <div className="profile-row">
          <div className="avatar">AB</div>
          <div className="profile-meta"><strong>Local workspace</strong><span>No account connected</span></div>
          <Settings2 size={17} className="muted-icon" />
        </div>
      </aside>

      <section className="main-area">
        <header className="topbar">
          <div className="breadcrumb"><span>Workspace</span><span className="crumb-slash">/</span><strong>{view === "translate" ? "Translator" : view === "library" ? "Sign library" : "Model lab"}</strong></div>
          <div className="topbar-actions">
            <span className="environment-chip"><span className="pulse-dot" /> PROTOTYPE ENVIRONMENT</span>
            <button className="icon-button" title="Privacy details" onClick={() => setNotice("Raw camera video is not saved or uploaded by this prototype. Only labeled 63-value landmark vectors are stored locally when you press Capture sample.")}><LockKeyhole size={17} /></button>
            <button className="help-button" onClick={() => setNotice("ASL Bridge is a research prototype. Landmark detection is not sign recognition; a trained classifier and ASL-reviewed clips are still needed.")}><CircleHelp size={16} /><span>Help</span></button>
          </div>
        </header>

        <div className="page-content">
          <div className="page-heading">
            <div>
              <div className="eyebrow"><span className="eyebrow-line" /> AMERICAN SIGN LANGUAGE <span className="eyebrow-sep">·</span> AI WORKBENCH</div>
              <h1>{view === "translate" ? "Conversation, seen differently." : view === "library" ? "A vocabulary with context." : "Train with intention."}</h1>
              <p className="page-subtitle">{view === "translate" ? "A focused workspace for hand tracking, sign practice, and model-backed recognition." : view === "library" ? "Explore the starter vocabulary and build practice sequences from supported gloss prompts." : "Collect labeled hand landmarks, export your dataset, then train and evaluate a classifier."}</p>
            </div>
            <div className="header-metrics">
              <div className="metric"><span className="metric-icon"><Zap size={15} /></span><div><strong>On-device vision</strong><small>Browser processed</small></div></div>
              <div className="metric"><span className="metric-icon purple"><Fingerprint size={15} /></span><div><strong>{samples.length.toLocaleString()} samples</strong><small>Local dataset</small></div></div>
            </div>
          </div>

          {notice && <div className="notice-banner" role="status"><Sparkles size={16} /><span>{notice}</span><button onClick={() => setNotice("")} aria-label="Dismiss message"><X size={15} /></button></div>}

          {view === "translate" && <>
            <div className="direction-switch" role="tablist" aria-label="Translation direction">
              <button className={direction === "sign-to-text" ? "selected" : ""} onClick={() => setDirection("sign-to-text")} role="tab" aria-selected={direction === "sign-to-text"}><span className="direction-icon"><Hand size={16} /></span><span><strong>Sign <ArrowDownRight size={13} /> Text</strong><small>Live recognition</small></span></button>
              <button className={direction === "text-to-sign" ? "selected" : ""} onClick={() => setDirection("text-to-sign")} role="tab" aria-selected={direction === "text-to-sign"}><span className="direction-icon direction-icon-purple"><MessageSquareText size={16} /></span><span><strong>Text <ArrowDownLeft size={13} /> Sign</strong><small>Gloss practice</small></span></button>
              <div className="direction-helper"><ArrowLeftRight size={15} /><span>Bidirectional workspace</span></div>
            </div>

            {direction === "sign-to-text" ? <div className="studio-grid">
              <div className="column-stack">
                <section className="panel camera-panel">
                  <div className="panel-head">
                    <div className="panel-title"><span className="panel-icon"><ScanLine size={17} /></span><div><h2>Live capture</h2><p>Real-time hand landmark tracking</p></div></div>
                    <span className={`status-pill ${cameraOn ? "status-active" : "status-idle"}`}><span />{cameraOn ? "CAMERA ACTIVE" : "CAMERA OFF"}</span>
                  </div>
                  <div className={`camera-stage ${cameraOn ? "camera-running" : ""}`}>
                    <video ref={videoRef} className="camera-video" playsInline muted aria-label="Live camera preview" />
                    <canvas ref={canvasRef} className="landmark-canvas" aria-hidden="true" />
                    {!cameraOn && <div className="camera-placeholder">
                      <div className="camera-orbit orbit-one" /><div className="camera-orbit orbit-two" />
                      <div className="camera-placeholder-icon"><Hand size={34} strokeWidth={1.5} /></div>
                      <strong>Ready when you are</strong>
                      <span>Position one hand in frame to start tracking landmarks.</span>
                      <div className="camera-permissions"><LockKeyhole size={12} /> Camera access is requested only when you start.</div>
                    </div>}
                    <div className="camera-overlay-top"><span className="camera-tag"><span className={cameraOn ? "camera-dot on" : "camera-dot"} /> {cameraOn ? "LIVE FEED" : "CAMERA FEED"}</span><span className="camera-resolution">{cameraOn ? "LANDMARK MODE" : "1280 × 720 TARGET"}</span></div>
                    {cameraOn && <div className="camera-overlay-bottom"><span><span className="live-bars"><i /><i /><i /><i /></span> Tracking</span><span>{handsDetected} {handsDetected === 1 ? "hand" : "hands"} detected</span></div>}
                  </div>
                  {cameraError && <div className="inline-error" role="alert">{cameraError}</div>}
                  <div className="camera-controls">
                    {!cameraOn ? <button className="button-primary" onClick={() => void startCamera()} disabled={cameraLoading}>{cameraLoading ? <span className="spinner" /> : <Video size={16} />}{cameraLoading ? "Starting tracker…" : "Start camera"}</button> : <button className="button-secondary danger-soft" onClick={stopCamera}><Square size={14} fill="currentColor" /> Stop camera</button>}
                    <div className="control-hint"><span className={`small-status ${detectorReady ? "good" : ""}`} />{detectorReady ? "Hand landmarks ready" : "Landmark detector idle"}</div>
                    <button className="text-button" onClick={() => setNotice("Hand landmarks are 21 points per detected hand. They describe pose only; they do not tell us the ASL sign without a trained classifier.")}><CircleHelp size={14} /> What is tracked?</button>
                  </div>
                </section>

                <section className="panel dataset-panel">
                  <div className="panel-head compact-head"><div className="panel-title"><span className="panel-icon violet-bg"><Fingerprint size={17} /></span><div><h2>Capture training sample</h2><p>Save the current normalized 21-point hand shape</p></div></div><span className="feature-badge">63 FEATURES</span></div>
                  <div className="capture-fields">
                    <label className="field-label">SIGN LABEL<input value={sampleLabel} onChange={(event) => setSampleLabel(event.target.value)} placeholder="e.g. HELLO or A" maxLength={50} /></label>
                    <label className="field-label">SIGNER / SESSION ID<input value={signerId} onChange={(event) => setSignerId(event.target.value)} placeholder="e.g. signer-01" maxLength={80} /></label>
                  </div>
                  <div className="capture-footer"><div className="capture-count"><span className="capture-count-icon"><Layers3 size={15} /></span><span><strong>{samples.length} samples captured</strong><small>{labelCounts.length} unique labels in this browser</small></span></div><div className="capture-actions"><button className="button-secondary" onClick={captureSample} disabled={!cameraOn}><Plus size={15} /> Capture sample</button><button className="button-secondary icon-only-button" onClick={exportSamples} title="Export JSONL" disabled={!samples.length}><Download size={15} /></button></div></div>
                  {sampleNotice && <div className="sample-notice" role="status">{sampleNotice}</div>}
                  <div className="privacy-inline"><ShieldCheck size={14} /><span>Only landmark coordinates and your label are saved locally. No image or video is recorded.</span></div>
                </section>
              </div>

              <div className="column-stack">
                <section className="panel output-panel">
                  <div className="panel-head"><div className="panel-title"><span className="panel-icon mint-bg"><AudioLines size={17} /></span><div><h2>Recognition output</h2><p>Classifier predictions appear here</p></div></div><span className="model-chip">{apiEnabled ? apiStatusLabel(apiStatus) : "NO MODEL"}</span></div>
                  <div className="recognition-empty">
                    <div className="recognition-emblem"><Activity size={23} /></div>
                    <span className="overline-small">LATEST PREDICTION</span>
                    {prediction ? <><strong className="prediction-label">{prediction.status === "uncertain" ? "Uncertain sign" : prediction.label}</strong><span className="prediction-confidence">{Math.round(prediction.confidence * 100)}% model confidence</span></> : <><strong className="empty-prediction">Waiting for a trained model</strong><p>Hand tracking is active, but no ASL classifier is configured yet.</p></>}
                    <div className="recognition-divider" />
                    <div className="output-meta"><span>Input</span><strong>{handsDetected ? `${handsDetected} detected ${handsDetected === 1 ? "hand" : "hands"}` : "No hand detected"}</strong></div>
                    <div className="output-meta"><span>Classification</span><strong>{apiStatus === "ready" && apiEnabled ? "Model connected" : "Not configured"}</strong></div>
                    {prediction && <button className="button-primary full-width" onClick={addPredictionToTranscript}><Plus size={15} /> Add to transcript</button>}
                  </div>
                  <div className="api-settings">
                    <div><strong>Connect local model API</strong><small>Sends landmark vectors only, never camera frames.</small></div>
                    <button className={`toggle ${apiEnabled ? "toggle-on" : ""}`} role="switch" aria-checked={apiEnabled} aria-label="Enable local model API" onClick={() => void toggleApi()}><span /></button>
                  </div>
                  {apiEnabled && <div className={`api-message ${apiStatus}`}><span className="api-status-dot" />{apiStatus === "checking" ? "Checking API health…" : apiStatus === "ready" ? "API connected. Classifier is available." : apiStatus === "no-model" ? "API is online, but no trained model was found. Follow the README training steps." : "API is offline. Start services/api and try again."}</div>}
                </section>

                <section className="panel transcript-panel">
                  <div className="panel-head"><div className="panel-title"><span className="panel-icon purple-bg"><Clipboard size={17} /></span><div><h2>Session transcript</h2><p>Build a phrase from reviewed predictions</p></div></div><button className="subtle-action" onClick={() => setTranscript([])} disabled={!transcript.length}>Clear</button></div>
                  <div className="transcript-body">{transcript.length ? <p>{transcript.join(" ")}</p> : <div className="transcript-empty"><MessageSquareText size={19} /><span>Your accepted predictions will appear here.</span></div>}</div>
                  <div className="transcript-footer"><span>{transcript.length} accepted {transcript.length === 1 ? "token" : "tokens"}</span><div><button className="round-action" title="Read transcript aloud" onClick={() => speakText(transcript.join(" "))} disabled={!transcript.length}><Volume2 size={15} /></button><button className="round-action" title="Copy transcript" onClick={() => { void navigator.clipboard?.writeText(transcript.join(" ")); setNotice("Transcript copied if clipboard permissions are available."); }} disabled={!transcript.length}><Clipboard size={15} /></button></div></div>
                </section>

                <div className="callout-note"><span className="callout-icon"><CircleHelp size={17} /></span><p><strong>Recognition vs. tracking</strong> A visible hand skeleton means the camera pipeline works. It does not mean an ASL sign has been identified.</p></div>
              </div>
            </div> : <div className="text-sign-layout">
              <section className="panel phrase-editor">
                <div className="panel-head"><div className="panel-title"><span className="panel-icon purple-bg"><Keyboard size={17} /></span><div><h2>English input</h2><p>Enter a phrase to create a sign-gloss practice queue</p></div></div><span className="feature-badge">PRACTICE MODE</span></div>
                <label className="sr-only" htmlFor="english-phrase">English phrase</label>
                <textarea id="english-phrase" value={phraseInput} onChange={(event) => { setPhraseInput(event.target.value); setPracticeBuilt(false); }} placeholder="Type a phrase to practice…" rows={5} />
                <div className="phrase-examples"><span>Try:</span>{["Hello", "Thank you", "How are you?", "I need help"].map((example) => <button key={example} onClick={() => { setPhraseInput(example); setPracticeBuilt(false); }}>{example}</button>)}</div>
                <div className="phrase-editor-footer"><span><LockKeyhole size={13} /> Text is processed locally in this prototype</span><button className="button-primary" onClick={() => setPracticeBuilt(true)}><Sparkles size={15} /> Build practice queue</button></div>
              </section>

              <section className="panel sequence-panel">
                <div className="panel-head"><div className="panel-title"><span className="panel-icon mint-bg"><Layers3 size={17} /></span><div><h2>Practice sequence</h2><p>ASL gloss prompts for supported entries</p></div></div><span className="status-pill status-idle"><span /> GLOSS PREVIEW</span></div>
                {practiceBuilt ? <>
                  {builtSequence.tokens.length ? <div className="gloss-sequence">{builtSequence.tokens.map((token, index) => <div className={`gloss-token ${token.known ? "known" : "unknown"}`} key={`${token.text}-${index}`}><span className="gloss-token-number">{String(index + 1).padStart(2, "0")}</span><strong>{token.text}</strong><small>{token.known ? "Gloss prompt" : "Needs dictionary review"}</small></div>)}</div> : <div className="blank-state"><MessageSquareText size={20} /><strong>Add some text first</strong><span>Enter an English phrase above to build the queue.</span></div>}
                  {builtSequence.exact && <div className="entry-detail"><span className="entry-detail-icon"><BookOpen size={16} /></span><div><strong>{builtSequence.exact.phrase}</strong><p>{builtSequence.exact.note}</p></div></div>}
                  {builtSequence.tokens.length > 0 && <div className="practice-video-placeholder"><div className="placeholder-art"><Hand size={26} /><span>ASL</span></div><div><strong>Reviewed signing demonstration</strong><p>No vetted video is bundled yet. Add a licensed, ASL-reviewed clip to the entry to enable playback.</p></div><span className="asset-status">ASSET NEEDED</span></div>}
                </> : <div className="blank-state"><div className="blank-state-icon"><ArrowDownLeft size={22} /></div><strong>Your practice queue starts here</strong><span>Choose an example or enter a phrase, then build the queue.</span></div>}
                <div className="sequence-footnote"><ShieldCheck size={14} /><span>This is a gloss and vocabulary aid, not a complete English-to-ASL translation. Word order, facial expression, movement, and spatial grammar matter.</span></div>
              </section>
            </div>}
          </>}

          {view === "library" && <div className="library-view">
            <div className="library-toolbar"><div className="search-field"><Search size={17} /><input value={libraryQuery} onChange={(event) => setLibraryQuery(event.target.value)} placeholder="Search phrases, glosses, notes…" aria-label="Search sign library" />{libraryQuery && <button onClick={() => setLibraryQuery("")} aria-label="Clear search"><X size={14} /></button>}</div><label className="category-select"><span>Category</span><select value={libraryCategory} onChange={(event) => setLibraryCategory(event.target.value)}><option>All</option><option>Conversation</option><option>Everyday</option><option>Help</option></select><ChevronDown size={14} /></label></div>
            <div className="library-summary"><span>{filteredLibrary.length} starter entries</span><span>Gloss prompts only · Add reviewed media before production use</span></div>
            <div className="library-grid">{filteredLibrary.map((entry, index) => <article className="panel library-card" key={entry.id}><div className="library-card-top"><div className={`library-glyph glyph-${index % 4}`}><Hand size={26} strokeWidth={1.7} /><span>{initials(entry.phrase)}</span></div><span className="library-category">{entry.category}</span></div><div className="library-entry-text"><h3>{entry.phrase}</h3><p>{entry.note}</p></div><div className="library-gloss"><span>GLOSS CUE</span><div>{entry.gloss.map((word) => <kbd key={word}>{word}</kbd>)}</div></div><div className="library-card-footer"><span className="asset-needed"><span /> No reviewed clip</span><button onClick={() => openPractice(entry)}>Practice phrase <ArrowDownRight size={14} /></button></div></article>)}</div>
            <div className="library-disclaimer"><CircleHelp size={17} /><p><strong>Contribute responsibly.</strong> Gloss cues can simplify ASL grammar. Have a Deaf ASL user or qualified reviewer verify each entry and any video before treating this library as learning guidance.</p></div>
          </div>}

          {view === "model" && <div className="model-view">
            <div className="model-overview-grid"><div className="panel model-stat-panel"><span className="stat-label">LANDMARK SAMPLES</span><div className="stat-number">{samples.length.toLocaleString()}<span>samples</span></div><p>Saved in this browser's local storage.</p><div className="stat-icon mint-bg"><Layers3 size={19} /></div></div><div className="panel model-stat-panel"><span className="stat-label">LABEL CLASSES</span><div className="stat-number">{labelCounts.length}<span>classes</span></div><p>Unique labels captured for training.</p><div className="stat-icon purple-bg"><Fingerprint size={19} /></div></div><div className="panel model-stat-panel"><span className="stat-label">CLASSIFIER STATUS</span><div className="stat-number stat-text">{apiStatus === "ready" && apiEnabled ? "Connected" : "Not trained"}</div><p>A saved model is required for prediction.</p><div className="stat-icon amber-bg"><Gauge size={19} /></div></div></div>
            <div className="model-lab-grid"><section className="panel model-labels-panel"><div className="panel-head"><div className="panel-title"><span className="panel-icon mint-bg"><Layers3 size={17} /></span><div><h2>Local dataset labels</h2><p>Sample balance by class</p></div></div><button className="subtle-action" onClick={exportSamples} disabled={!samples.length}><FileDown size={14} /> Export JSONL</button></div>{labelCounts.length ? <div className="label-list">{labelCounts.map(([label, count]) => <div className="label-row" key={label}><span className="label-name">{label}</span><div className="label-bar"><span style={{ width: `${Math.max(4, count / Math.max(...labelCounts.map((entry) => entry[1])) * 100)}%` }} /></div><span className="label-count">{count}</span></div>)}</div> : <div className="blank-state small-blank"><div className="blank-state-icon"><Fingerprint size={20} /></div><strong>No samples yet</strong><span>Open Translator → Sign to Text, start the camera, enter a label, and capture repeated examples.</span><button className="button-secondary" onClick={() => { setView("translate"); setDirection("sign-to-text"); }}>Open data collector <ArrowDownRight size={14} /></button></div>}<div className="model-panel-actions"><button className="button-secondary" onClick={clearSamples} disabled={!samples.length}><X size={14} /> Clear local data</button><button className="button-primary" onClick={exportSamples} disabled={!samples.length}><Download size={14} /> Download JSONL</button></div></section>
              <section className="panel training-guide-panel"><div className="panel-head"><div className="panel-title"><span className="panel-icon purple-bg"><Activity size={17} /></span><div><h2>Training workflow</h2><p>From landmarks to a measurable baseline</p></div></div></div><div className="workflow-list"><div className="workflow-step"><span className="step-no">01</span><div><strong>Capture balanced samples</strong><p>Record multiple examples for every label, across separate sessions and signers where possible.</p><span className="step-meta">DATA COLLECTION</span></div></div><div className="workflow-step"><span className="step-no">02</span><div><strong>Export your JSONL dataset</strong><p>Download samples and place the file at <code>data/samples.jsonl</code>.</p><span className="step-meta">DATA VERSIONING</span></div></div><div className="workflow-step"><span className="step-no">03</span><div><strong>Train the Random Forest</strong><p>Run <code>python ml/train.py</code> from the project root to create the baseline artifact.</p><span className="step-meta">MODEL TRAINING</span></div></div><div className="workflow-step"><span className="step-no">04</span><div><strong>Run the local API</strong><p>Start FastAPI on port 8000, then enable the model connection in the Translator.</p><span className="step-meta">LIVE INFERENCE</span></div></div></div><a className="docs-link" href="https://github.com/raghav19sh/sign-lang-translator" target="_blank" rel="noreferrer">Open project repository <ExternalLink size={14} /></a></section></div>
            <div className="model-warning"><CircleHelp size={18} /><div><strong>Model quality depends on the dataset.</strong><p>A Random Forest trained on your own captures is only a baseline. Split by signer when possible, evaluate on unseen signers, inspect per-class metrics, and do not describe alphabet classification as full ASL translation. Dynamic signs such as J and Z require temporal data.</p></div></div>
          </div>}

          <footer className="page-footer"><div><span className="footer-dot" /> SYSTEM STATUS <strong>{cameraOn ? "LANDMARK TRACKING" : "READY"}</strong><span className="footer-divider">/</span><span>ASL BRIDGE v1.0</span></div><div><span>Built for learning and research</span><span className="footer-divider">·</span><button onClick={() => setNotice("This prototype is not a substitute for a qualified ASL interpreter or emergency communication service.")}>Limitations</button><span className="footer-divider">·</span><button onClick={() => setNotice("Privacy note: camera video remains in the browser. Landmark samples are stored locally only after you press Capture sample.")}>Privacy</button></div></footer>
        </div>
      </section>
    </main>
  );
}

function apiStatusLabel(status: "off" | "checking" | "ready" | "no-model" | "offline") {
  if (status === "ready") return "MODEL READY";
  if (status === "checking") return "CHECKING";
  if (status === "offline") return "API OFFLINE";
  if (status === "no-model") return "NO MODEL";
  return "NO MODEL";
}
