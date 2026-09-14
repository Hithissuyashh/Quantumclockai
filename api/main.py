import asyncio
from collections import deque
from threading import Lock

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from simulator.clock import QuantumClock


app = FastAPI(
    title="QuantumClockAI API",
    description="Real-time quantum clock digital twin backend",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Simulation state
# ---------------------------------------------------------------------------

clock = QuantumClock(seed=42)

clock_lock = Lock()

MAX_HISTORY = 1000

history = deque(maxlen=MAX_HISTORY)

# Active WebSocket subscribers.
subscribers: set[WebSocket] = set()

# Latest serialized telemetry sample.
latest_sample = None

# Background simulation task.
simulation_task = None


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def serialize_result(result):
    """
    Convert simulator output into JSON-safe telemetry.

    Only exposes values actually produced by the simulator.
    No telemetry is fabricated or synthesized here.
    """

    ai = result.get("ai", {})

    return {
        "time": float(result["time"]),
        "atomic_frequency": float(result["atomic_frequency"]),
        "true_detuning": float(result["true_detuning"]),
        "measured_frequency": float(result["measured_frequency"]),
        "measured_offset": float(result["measured_offset"]),
        "estimated_offset": float(result["estimated_offset"]),
        "servo_correction": float(result["servo_correction"]),
        "corrected_offset": float(result["corrected_offset"]),
        "fractional_frequency": float(result["fractional_frequency"]),

        # Quantum interrogation telemetry. These values come from the
        # configured simulator components; unavailable discriminator
        # internals remain explicitly null rather than being fabricated.
        "probe_offset": (
            None
            if result.get("probe_offset") is None
            else float(result["probe_offset"])
        ),
        "interrogation_time": (
            None
            if result.get("interrogation_time") is None
            else float(result["interrogation_time"])
        ),
        "discriminator": (
            None
            if result.get("discriminator") is None
            else float(result["discriminator"])
        ),

        "excitation_probability_plus": float(
            result["excitation_probability_plus"]
        ),
        "excitation_probability_minus": float(
            result["excitation_probability_minus"]
        ),

        "environment": {
            key: float(value)
            for key, value in result["environment"].items()
        },

        "noise": {
            key: float(value)
            for key, value in result["noise"].items()
        },

        "ai": {
            "prediction": (
                None
                if ai.get("prediction") is None
                else float(ai["prediction"])
            ),
            "extra_correction": float(
                ai.get("extra_correction", 0.0)
            ),
            "gain": float(
                ai.get("gain", 0.0)
            ),
            "correction_limit": float(
                ai.get("correction_limit", 0.0)
            ),
            "ready": bool(
                ai.get("ready", False)
            ),
            "history_length": int(
                ai.get("history_length", 0)
            ),
            "prediction_count": int(
                ai.get("prediction_count", 0)
            ),
            "required_history": int(
                ai.get("required_history", 128)
            ),
            "uncertainty": (
                None
                if ai.get("uncertainty") is None
                else float(ai["uncertainty"])
            ),
            "uncertainty_calibrated": bool(
                ai.get("uncertainty_calibrated", False)
            ),
            "interval_low": (
                None
                if ai.get("interval_low") is None
                else float(ai["interval_low"])
            ),
            "interval_high": (
                None
                if ai.get("interval_high") is None
                else float(ai["interval_high"])
            ),
        },
    }


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def generate_sample():
    """
    Advance the simulator exactly once and store the resulting telemetry.
    """

    global latest_sample

    with clock_lock:
        result = clock.step()
        serialized = serialize_result(result)

        history.append(serialized)
        latest_sample = serialized

        return serialized


async def broadcast_sample(sample):
    """
    Send one telemetry sample to every connected WebSocket client.

    A failed connection is removed without affecting other subscribers.
    """

    if not subscribers:
        return

    disconnected = set()

    for websocket in list(subscribers):
        try:
            await websocket.send_json(sample)
        except Exception:
            disconnected.add(websocket)

    for websocket in disconnected:
        subscribers.discard(websocket)


async def simulation_loop():
    """
    Single authoritative simulation loop.

    This is the only continuous process that advances clock.step().
    Every WebSocket subscriber receives the same generated sample.
    """

    while True:
        sample = generate_sample()

        await broadcast_sample(sample)

        await asyncio.sleep(1.0)


# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    global simulation_task

    simulation_task = asyncio.create_task(
        simulation_loop()
    )


@app.on_event("shutdown")
async def shutdown_event():
    global simulation_task

    if simulation_task is not None:
        simulation_task.cancel()

        try:
            await simulation_task
        except asyncio.CancelledError:
            pass

        simulation_task = None


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "name": "QuantumClockAI",
        "status": "online",
        "service": "quantum clock digital twin",
        "version": "1.0.0",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "QuantumClockAI API",
    }


@app.get("/status")
def status():
    with clock_lock:
        return {
            "status": "running",
            "time": float(clock.environment.time),
            "history_length": len(history),
            "connected_clients": len(subscribers),
        }


@app.post("/step")
def step():
    """
    Manually advance the simulator by one step.

    The continuous simulation loop remains the authoritative
    source for normal real-time operation.
    """

    return generate_sample()


@app.get("/history")
def get_history():
    with clock_lock:
        return {
            "count": len(history),
            "data": list(history),
        }


@app.post("/reset")
def reset():
    global clock
    global latest_sample

    with clock_lock:
        clock.reset()
        history.clear()
        latest_sample = None

        return {
            "status": "reset",
            "time": 0.0,
        }


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

@app.websocket("/ws/clock")
async def clock_stream(websocket: WebSocket):
    await websocket.accept()

    subscribers.add(websocket)

    try:
        # Send the latest available sample immediately.
        if latest_sample is not None:
            await websocket.send_json(latest_sample)

        # Keep this connection alive.
        while True:
            await websocket.receive_text()

    except WebSocketDisconnect:
        pass

    except Exception:
        pass

    finally:
        subscribers.discard(websocket)