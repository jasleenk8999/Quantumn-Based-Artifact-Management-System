from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pennylane as qml
import os
from rtree import index
import itertools
from collections import deque
import heapq
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, LSTM
import google.generativeai as genai

# ---------------- CONFIGURE AI AGENT ---------------- #
# Set your Gemini API key here or in your environment variables
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyBPrA44wr4Cl4vtzN-XCa-hLHAiPsTURRw")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

app = Flask(__name__)

# ---------------- ADVANCED DATA STRUCTURES ---------------- #

class MaxHeap:
    """A priority queue that guarantees O(log N) access to the maximum element."""
    def __init__(self):
        self.heap = []
    
    def push(self, score, item_id):
        # Invert score to simulate Max-Heap using Python's Min-Heap
        heapq.heappush(self.heap, (-score, item_id))
        
    def pop(self):
        if self.heap:
            score, item_id = heapq.heappop(self.heap)
            return item_id, -score
        return None, None
        
    def __len__(self):
        return len(self.heap)


class DoubleEndedPriorityQueue:
    """
    A Double-Ended Priority Queue (DEPQ) that allows O(log N) extraction 
    of BOTH the maximum (highest risk) and minimum (healthiest) elements.
    Uses lazy-deletion via shared mutable state to keep heaps synchronized.
    """
    def __init__(self):
        self.min_heap = []
        self.max_heap = []
        self.entry_finder = {}
        self.counter = itertools.count()

    def update(self, item_id, score):
        # Mark existing entry as removed (lazy deletion)
        if item_id in self.entry_finder:
            self.entry_finder[item_id][0] = False
            
        count = next(self.counter)
        is_active = [True] # Shared mutable flag
        
        # Min-Heap entry
        heapq.heappush(self.min_heap, [score, count, item_id, is_active])
        # Max-Heap entry
        heapq.heappush(self.max_heap, [-score, count, item_id, is_active])
        
        self.entry_finder[item_id] = is_active

    def _clean_max_heap(self):
        while self.max_heap and not self.max_heap[0][3][0]:
            heapq.heappop(self.max_heap)
            
    def get_max(self):
        self._clean_max_heap()
        if self.max_heap:
            return self.max_heap[0][2], -self.max_heap[0][0]
        return None, None
        
    def _clean_min_heap(self):
        while self.min_heap and not self.min_heap[0][3][0]:
            heapq.heappop(self.min_heap)
            
    def get_min(self):
        self._clean_min_heap()
        if self.min_heap:
            return self.min_heap[0][2], self.min_heap[0][0]
        return None, None


# ---------------- GLOBAL STORAGE ---------------- #

engine_scores = {}  # Replaced maintenance_queue list with dict for O(N) ranking
global_depq = DoubleEndedPriorityQueue()  # O(log N) min/max tracking
history_vault = {}
score_history = {}

# ---------------- LOAD MODELS ---------------- #

with open("models/quantum_weights.pkl", "rb") as f:
    weights = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# --- Workaround for Keras 3 to Keras 2 Backward Compatibility ---
_original_dense_init = Dense.__init__
def _patched_dense_init(self, *args, **kwargs):
    kwargs.pop('quantization_config', None)
    _original_dense_init(self, *args, **kwargs)
Dense.__init__ = _patched_dense_init

_original_lstm_init = LSTM.__init__
def _patched_lstm_init(self, *args, **kwargs):
    kwargs.pop('quantization_config', None)
    _original_lstm_init(self, *args, **kwargs)
LSTM.__init__ = _patched_lstm_init
# ----------------------------------------------------------------

future_model = load_model("models/future_score_model.keras")

# ---------------- QUANTUM DEVICE ---------------- #

dev = qml.device("default.qubit", wires=4)

@qml.qnode(dev)
def q_circuit(weights, inputs):
    for i in range(4):
        qml.RY(inputs[i] * np.pi, wires=i)

    for l in range(2):
        for i in range(4):
            qml.CNOT(wires=[i, (i+1)%4])
        for i in range(4):
            qml.Rot(weights[l,i,0], weights[l,i,1], weights[l,i,2], wires=i)

    return qml.expval(qml.PauliZ(0))

# ---------------- R-TREE ---------------- #

p = index.Property()
idx = index.Index(properties=p)

hangar_data = {
    1: (28.5562, 77.1000, "Indira Gandhi International Airport (DEL)"),
    2: (19.0896, 72.8656, "Chhatrapati Shivaji Maharaj International Airport (BOM)"),
    3: (13.1986, 77.7066, "Kempegowda International Airport (BLR)"),
    4: (17.2403, 78.4294, "Rajiv Gandhi International Airport (HYD)"),
    5: (22.6547, 88.4467, "Netaji Subhas Chandra Bose International Airport (CCU)"),
    6: (13.0827, 80.2707, "Chennai International Airport (MAA)"),
    7: (23.0726, 72.6347, "Sardar Vallabhbhai Patel International Airport (AMD)"),
    8: (15.3808, 73.8314, "Goa International Airport (GOI)"),
    9: (11.1368, 77.0420, "Coimbatore International Airport (CJB)"),
    10: (8.4821, 76.9201, "Trivandrum International Airport (TRV)"),
    11: (18.5821, 73.9197, "Pune International Airport (PNQ)")
}

for h_id, (lat, lon, name) in hangar_data.items():
    idx.insert(h_id, (lat, lon, lat, lon))

# ---------------- HELPER FUNCTIONS ---------------- #

def smooth_score(engine_id, new_score, alpha=0.6):
    if engine_id not in score_history:
        # Deque provides O(1) time complexity for appending and automatically removes oldest entries
        score_history[engine_id] = deque(maxlen=10)

    hist = score_history[engine_id]

    if len(hist) == 0:
        smoothed = new_score
    else:
        smoothed = alpha * new_score + (1 - alpha) * hist[-1]

    hist.append(smoothed)

    return smoothed


def get_trend(engine_id):
    hist = score_history.get(engine_id, [])

    if len(hist) < 3:
        return "Stable"

    if hist[-1] > hist[-2] > hist[-3]:
        return "Increasing Risk 📈"
    elif hist[-1] < hist[-2] < hist[-3]:
        return "Improving 📉"
    else:
        return "Stable"


def get_sequence(engine_id, window=3):
    data = history_vault.get(engine_id, [])

    if len(data) < 3:
        return None

    # Convert deque to list to allow slicing
    recent_data = list(data)[-window:]
    # Pad with the oldest available data point if we have less than 3
    while len(recent_data) < window:
        recent_data.insert(0, recent_data[0])

    seq = np.array(recent_data)
    return seq.reshape(1, window, 4)

# ---------------- ROUTES ---------------- #

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict-page")
def predict_page():
    return render_template("predict.html")

# ---------------- MAIN API ---------------- #

@app.route("/predict", methods=["POST"])
def predict():
    global engine_scores

    data = request.json

    engine_id = data.get("engine_id", "ENG-1")
    
    try:
        curr_lat = float(data.get("lat") if data.get("lat") is not None else 18.5204)
        curr_lon = float(data.get("lon") if data.get("lon") is not None else 73.8567)
    except (ValueError, TypeError):
        curr_lat, curr_lon = 18.5204, 73.8567

    # ✅ SAFE REAL VALUES INPUT (Handles empty strings from frontend)
    try:
        sensors = np.array([[
            float(data.get("s11") if data.get("s11") else 0.0),
            float(data.get("s12") if data.get("s12") else 0.0),
            float(data.get("s13") if data.get("s13") else 0.0),
            float(data.get("s15") if data.get("s15") else 0.0)
        ]])
    except (ValueError, TypeError):
        sensors = np.array([[0.0, 0.0, 0.0, 0.0]])

    # -------- STORE HISTORY -------- #
    if engine_id not in history_vault:
        # Prevent memory leaks over time; bounds history to last 50 entries
        history_vault[engine_id] = deque(maxlen=50)

    history_vault[engine_id].append(sensors[0].tolist())

    # -------- SCALE -------- #
    scaled = scaler.transform(sensors)

    # -------- QUANTUM PREDICTION -------- #
    raw_score = float(q_circuit(weights, scaled[0]))
    score = smooth_score(engine_id, raw_score)

    # -------- STATUS -------- #
    if score > 0.35:
        status = "⚠ High Failure Risk"
    elif score > -0.1:
        status = "🟡 Moderate Risk"
    else:
        status = "✅ Engine Healthy"

    # -------- FUTURE PREDICTION -------- #
    sequence = get_sequence(engine_id)
    future_health = None
    future_status = "Not enough data"

    if sequence is not None:
        window = sequence.shape[1]
        seq_scaled = scaler.transform(sequence.reshape(-1, 4)).reshape(1, window, 4)

        future_raw = float(future_model.predict(seq_scaled, verbose=0)[0][0])
        
        future_health = smooth_score(engine_id + "_future", future_raw)

        if future_health < 0.3:
            future_status = "⚠ Future Failure Risk"
        elif future_health < 0.6:
            future_status = "🟡 Future Moderate Risk"
        else:
            future_status = "✅ Future Healthy"
        # Enforce Rule: Engines cannot magically heal over time
        if score > 0.35:
            future_status = "⚠ Future Failure Risk"
            if future_health >= 0.3:
                future_health = min(future_health, 0.29)
        elif score > -0.1:
            if future_status == "✅ Future Healthy":
                future_status = "🟡 Future Moderate Risk"
                if future_health >= 0.6:
                    future_health = min(future_health, 0.59)

    # -------- PRIORITY RANKING -------- #
    engine_scores[engine_id] = score
    global_depq.update(engine_id, score)

    depq_max_eid, depq_max_score = global_depq.get_max()
    depq_min_eid, depq_min_score = global_depq.get_min()

    # Calculate rank in O(N) time instead of O(N log N)
    # Rank is determined by how many engines have a higher risk score
    # In case of tie, we fall back to alphabetical engine_id comparison to be stable
    rank = sum(1 for eid, s in engine_scores.items() if s > score or (s == score and eid < engine_id)) + 1

    # -------- NEAREST AIRPORT -------- #
    nearest = list(idx.nearest((curr_lat, curr_lon, curr_lat, curr_lon), 1))
    hangar_lat, hangar_lon, hangar_name = hangar_data[nearest[0]]

    # -------- AUTO DECISION AGENT -------- #
    trend_val = get_trend(engine_id)
    
    def generate_decision(score, future_health, trend, rank, hangar_name):
        if not GEMINI_API_KEY:
            return "⚠️ Real AI Agent is offline. Please paste your GEMINI_API_KEY in main.py (line 10) to activate the LLM decision engine."

        prompt = f"""
        You are an expert AI Aircraft Maintenance Director. Given the following telemetry for an aircraft engine, provide a short, dynamic, 2-sentence operational decision.
        Do not use generic text, make it unique and authoritative. Be urgent if scores are critical.
        
        Current Quantum Risk Score: {score} (higher than 0.35 is critical, higher than -0.1 is warning)
        Future LSTM Health Score: {future_health} (lower than 0.3 is critical)
        Degradation Trend: {trend}
        Nearest Hangar: {hangar_name}
        Priority Queue Rank: {rank}
        
        Output ONLY the final decision text.
        """
        try:
            model = genai.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"🚨 AI Agent API Error: {str(e)}"

    decision_text = generate_decision(score, future_health, trend_val, rank, hangar_name)

    # -------- RESPONSE -------- #
    return jsonify({
        "status": status,
        "score": round(score, 4),
        "trend": trend_val,
        "decision": decision_text,

        "future_health": round(future_health, 4) if future_health is not None else None,
        "future_status": future_status,

        "priority": f"Priority Rank: {rank}",
        "hangar": hangar_name,
        "processed_at": [curr_lat, curr_lon],
        "history_version": len(history_vault[engine_id]),
        "hangar_coords": [hangar_lat, hangar_lon],
        
        "depq_max_engine": depq_max_eid,
        "depq_max_score": round(depq_max_score, 4) if depq_max_score is not None else None,
        "depq_min_engine": depq_min_eid,
        "depq_min_score": round(depq_min_score, 4) if depq_min_score is not None else None
    })

# ---------------- REPORT ---------------- #

@app.route("/report")
def report_page():
    # Priority Queue (Max-Heap) implementation using our formal MaxHeap class
    max_heap = MaxHeap()
    for eid, s in engine_scores.items():
        max_heap.push(s, eid)
        
    # Extracting items from heap maintains priority queue behavior
    sorted_queue = []
    while len(max_heap) > 0:
        eid, s = max_heap.pop()
        sorted_queue.append((-s, eid)) # Keep tuple format for template compatibility

    depq_max_eid, depq_max_score = global_depq.get_max()
    depq_min_eid, depq_min_score = global_depq.get_min()
    
    # Convert deques to lists for JSON serialization in the template graph
    score_hist_lists = {eid: list(hist) for eid, hist in score_history.items()}

    return render_template("report.html",
                           history=history_vault,
                           queue=sorted_queue,
                           depq_max=(depq_max_eid, depq_max_score),
                           depq_min=(depq_min_eid, depq_min_score),
                           score_history=score_hist_lists,
                           engine_scores=engine_scores)

# ---------------- RUN ---------------- #

if __name__ == "__main__":
    app.run(debug=True)
