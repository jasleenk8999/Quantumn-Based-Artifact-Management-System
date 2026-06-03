from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pennylane as qml
import heapq 
from rtree import index 

app = Flask(__name__)

# 🔥 Priority Queue (Global)
maintenance_queue = []

# 🔥 R-tree for nearest hangar search
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

# Insert hangars into R-tree
for h_id, (lat, lon, name) in hangar_data.items():
    idx.insert(h_id, (lat, lon, lat, lon))

# 🔥 History Storage
history_vault = {}

# Load model
with open("quantum_weights.pkl","rb") as f:
    weights = pickle.load(f)

with open("scaler.pkl","rb") as f:
    scaler = pickle.load(f)

# Quantum device
dev = qml.device("default.qubit", wires=4)

@qml.qnode(dev)
def q_circuit(weights, inputs):
    for i in range(4):
        qml.RY(inputs[i] * np.pi, wires=i)

    for l in range(2):
        for i in range(4):
            qml.CNOT(wires=[i,(i+1)%4])
        for i in range(4):
            qml.Rot(weights[l,i,0], weights[l,i,1], weights[l,i,2], wires=i)

    return qml.expval(qml.PauliZ(0))


# ---------------- ROUTES ---------------- #

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict-page")
def predict_page():
    return render_template("predict.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    
    curr_lat = float(data.get("lat", 18.5204)) 
    curr_lon = float(data.get("lon", 73.8567))

    engine_id = data.get("engine_id", "ENG-777")

    sensors = np.array([[
        float(data["s11"]),
        float(data["s12"]),
        float(data["s13"]),
        float(data["s15"])
    ]])

    # 🔥 Store history
    if engine_id not in history_vault:
        history_vault[engine_id] = []
    history_vault[engine_id].append(sensors.tolist())
    version_count = len(history_vault[engine_id])

    # 🔥 Model prediction
    scaled = scaler.transform(sensors)
    result = q_circuit(weights, scaled[0])
    score = float(result)

    # ---------------- FIXED LOGIC ---------------- #

    # Status
    if score > 0:
        status = "⚠ High Failure Risk"
    else:
        status = "✅ Engine Healthy"

    # 🔥 Remove duplicate engine entries
    global maintenance_queue
    maintenance_queue[:] = [item for item in maintenance_queue if item[1] != engine_id]
    heapq.heapify(maintenance_queue)

    # 🔥 Always push into queue (for report visibility)
    heapq.heappush(maintenance_queue, (-abs(score), engine_id))

    priority_rank = f"Priority Rank: {len(maintenance_queue)}"

    # 🔥 Find nearest hangar
    nearest = list(idx.nearest((curr_lat, curr_lon, curr_lat, curr_lon), 1))
    nearest_id = nearest[0]

    hangar_lat, hangar_lon, hangar_name = hangar_data[nearest_id]
    recommended_hangar = hangar_name

    # ---------------- RESPONSE ---------------- #

    return jsonify({
        "status": status,
        "score": score,
        "priority": priority_rank,
        "hangar": recommended_hangar,
        "history_version": version_count,
        "processed_at": [curr_lat, curr_lon],
        "hangar_coords": [hangar_lat, hangar_lon]
    })


@app.route("/report")
def report_page():
    # 🔥 Debug prints
    print("QUEUE:", maintenance_queue)
    print("HISTORY:", history_vault)

    # 🔥 Proper sorting (highest risk first)
    sorted_queue = sorted(maintenance_queue, key=lambda x: x[0])

    return render_template("report.html", 
                           history=history_vault, 
                           queue=sorted_queue)


# ---------------- RUN APP ---------------- #

if __name__ == "__main__":
    app.run(debug=True)

