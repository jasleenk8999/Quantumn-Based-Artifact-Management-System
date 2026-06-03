# ✈️ QCAMP: Quantum-Based Aircraft Maintenance Prediction System

**QCAMP** is an advanced, intelligent aircraft engine monitoring system that merges **Quantum Machine Learning (QML)**, **Long Short-Term Memory (LSTM) Networks**, and **Generative AI** to provide state-of-the-art predictive maintenance telemetry.

Built with a stunning, high-contrast "Quantum-Cyberpunk" aesthetic, the system provides real-time failure prediction, trend analysis, and intelligent logistics handling using strictly optimized spatial indexing and priority queueing structures.

---

## 🚀 Key Features

* **🔮 Quantum Risk Prediction:** Uses a 4-qubit Variational Quantum Circuit (VQC) built with **PennyLane** to extract non-linear risk signatures from live engine telemetry.
* **⏳ LSTM Future Health Prognosis:** Implements a sliding-window Keras LSTM to analyze time-series degradation and predict future engine health states.
* **🧠 AI Agent Decision Engine:** Integrates **Google Gemini AI 2.5** to generate authoritative, context-aware operational directives based on complex multi-model engine states.
* **⚡ $O(1)$ Fleet Criticality Tracking:** A custom **Double-Ended Priority Queue (DEPQ)** built over dual Min/Max Heaps ensures $O(1)$ lookup for the healthiest and most at-risk engines.
* **🗺️ Instant Hangar Logistics:** Utilizes **R-Tree spatial indexing** to instantly locate the nearest global MRO (Maintenance, Repair, and Overhaul) facility in $O(\log N)$ time.
* **📈 Real-Time Interactive Telemetry:** Employs **Chart.js** to dynamically render Fleet Quantum Timelines and sleek Horizontal "Health Bar" visualizations.

---

## 🛠️ Data Structures & Architecture

QCAMP is heavily optimized to run in real-time under high simulation loads:
* **R-Tree Index:** Instantly prunes coordinate distances to find nearby hangars without scanning the globe.
* **DEPQ (Min/Max Heaps):** Utilizes lazy-deletion heaps to track fleet extremes.
* **Collections Deque:** Maintains fixed sliding windows ($O(1)$ operations) for both the LSTM input stream and the UI rendering history.
* **Hash Maps:** Enforces constant-time $O(1)$ telemetry lookups by Engine ID.

---

## ⚙️ Technologies Used

| Category | Tools |
| :--- | :--- |
| **Backend** | Python, Flask, NumPy, Scikit-Learn |
| **Quantum & ML** | PennyLane (QNN), TensorFlow (Keras LSTM) |
| **Generative AI** | Google Gemini SDK |
| **Data Structures** | R-Tree, Double-Ended Priority Queue, Deque |
| **Frontend UI** | HTML5, CSS3 (Glassmorphism), JavaScript |
| **Data Visualization** | Chart.js |

---

## 📂 Project Structure

```text
QCAMP/
├── main.py                  # Main Flask Backend Server
├── models/                  # ML Models & Weights
│   ├── future_score_model.keras  # Trained LSTM Model
│   └── scaler.pkl                # Data Normalizer
├── quantum_files/           # QNN Weights
├── templates/               # UI Dashboard Components
│   ├── index.html           # Landing Page
│   ├── predict.html         # Form Input
│   └── report.html          # Visual Analytics Dashboard
├── static/                  # Assets
│   ├── style.css            # Cyberpunk Design Tokens
│   └── script.js            # Frontend Logic & Charting
└── README.md
```

---

## 🔑 Installation & Environment

Before running the local Flask server, ensure you configure your Generative AI Key. The system will look for this key to dynamically render the AI Decision blocks.

```bash
# Windows (PowerShell)
$env:GEMINI_API_KEY="your_api_key_here"

# Linux / Mac
export GEMINI_API_KEY="your_api_key_here"

# Run the server
python main.py
```

## 👨‍💻 Author
**Girish Nalkar** Specializing in AI, Quantum Computing, and Aviation Safety Systems.
