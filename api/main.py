import asyncio
from collections import deque
from threading import Lock

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from simulator.clock import QuantumClock


# ==========================================================
# APPLICATION
# ==========================================================

app = FastAPI(
    title="QuantumClockAI API",
    description="Real-time quantum clock digital twin backend",
    version="1.0.0",
)


# ==========================================================
# CORS
# ==========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================================
# CLOCK
# ==========================================================

clock = QuantumClock(
    seed=42
)

clock_lock = Lock()


# ==========================================================
# HISTORY
# ==========================================================

MAX_HISTORY = 1000

history = deque(
    maxlen=MAX_HISTORY
)


# ==========================================================
# SERIALIZATION
# ==========================================================
def serialize_result(result):
    """
    Convert simulator output into JSON-safe data.
    """

    return {
        "time": float(
            result["time"]
        ),

        "atomic_frequency": float(
            result["atomic_frequency"]
        ),

        "true_detuning": float(
            result["true_detuning"]
        ),

        "measured_frequency": float(
            result["measured_frequency"]
        ),

        "measured_offset": float(
            result["measured_offset"]
        ),

        "estimated_offset": float(
            result["estimated_offset"]
        ),

        "servo_correction": float(
            result["servo_correction"]
        ),

        "corrected_offset": float(
            result["corrected_offset"]
        ),

        "fractional_frequency": float(
            result["fractional_frequency"]
        ),

        "excitation_probability_plus": float(
            result[
                "excitation_probability_plus"
            ]
        ),

        "excitation_probability_minus": float(
            result[
                "excitation_probability_minus"
            ]
        ),

        "environment": {
            key: float(value)
            for key, value
            in result["environment"].items()
        },

        "noise": {
            key: float(value)
            for key, value
            in result["noise"].items()
        },

        # --------------------------------------------------
        # Transformer AI telemetry
        # --------------------------------------------------

        "ai": {

            "prediction": (
                None
                if result["ai"]["prediction"] is None
                else float(
                    result["ai"]["prediction"]
                )
            ),

            "extra_correction": float(
                result["ai"]["extra_correction"]
            ),

            "gain": float(
                result["ai"]["gain"]
            ),

            "correction_limit": float(
                result["ai"]["correction_limit"]
            ),

            "ready": bool(
                result["ai"]["ready"]
            ),

            "history_length": int(
                result["ai"]["history_length"]
            ),

            "prediction_count": int(
                result["ai"]["prediction_count"]
            ),
        },
    }
# ==========================================================
# ROOT
# ==========================================================

@app.get("/")
def root():

    return {
        "name": "QuantumClockAI",
        "status": "online",
        "service": "quantum clock digital twin",
        "version": "1.0.0",
    }


# ==========================================================
# STATUS
# ==========================================================

@app.get("/status")
def status():

    with clock_lock:

        return {
            "status": "running",
            "time": float(
                clock.environment.time
            ),
            "history_length": len(history),
        }


# ==========================================================
# STEP
# ==========================================================

@app.post("/step")
def step():

    with clock_lock:

        result = clock.step()

        serialized = (
            serialize_result(result)
        )

        history.append(
            serialized
        )

        return serialized


# ==========================================================
# HISTORY
# ==========================================================

@app.get("/history")
def get_history():

    with clock_lock:

        return {
            "count": len(history),
            "data": list(history),
        }


# ==========================================================
# RESET
# ==========================================================

@app.post("/reset")
def reset():

    global clock

    with clock_lock:

        clock.reset()

        history.clear()

        return {
            "status": "reset",
            "time": 0.0,
        }


# ==========================================================
# WEBSOCKET
# ==========================================================

@app.websocket("/ws/clock")
async def clock_stream(websocket: WebSocket):

    await websocket.accept()

    try:

        while True:

            with clock_lock:

                result = clock.step()

                serialized = (
                    serialize_result(result)
                )

                history.append(
                    serialized
                )

            await websocket.send_json(
                serialized
            )

            await asyncio.sleep(1.0)

    except WebSocketDisconnect:

        print(
            "Clock WebSocket disconnected."
        )