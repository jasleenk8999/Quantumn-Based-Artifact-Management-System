function goPredict(){
    window.location.href="/predict-page"
}

// COMBINED FUNCTION: Fetch location
function runPrediction() {
    let resBox = document.getElementById("result");
    resBox.style.display = "block";
    resBox.innerHTML = `
        <div class="spinner"></div>
        <h2 class="animate-fade">Fetching Location & Processing...</h2>
        <p class="animate-fade" style="color: #a8b7c8; font-size: 14px; margin-top: 10px;">Evaluating quantum state vectors...</p>
    `;

    if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(
            (position) => {
                executePrediction(position.coords.latitude, position.coords.longitude);
            },
            (error) => {
                console.warn("Location denied. Using default Pune.");
                executePrediction(18.5204, 73.8567);
            }
        );
    } else {
        executePrediction(18.5204, 73.8567);
    }
}

// CORE LOGIC
function executePrediction(lat, lon) {
    let s11 = document.getElementById("s11").value;
    let s12 = document.getElementById("s12").value;
    let s13 = document.getElementById("s13").value;
    let s15 = document.getElementById("s15").value;
    let engine_id = document.getElementById("engine_id") 
        ? document.getElementById("engine_id").value 
        : "ENG-777";

    fetch("/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            engine_id: engine_id,
            s11: s11,
            s12: s12,
            s13: s13,
            s15: s15,
            lat: lat,
            lon: lon
        })
    })
    .then(res => res.json())
    .then(data => {
        let resBox = document.getElementById("result");

        // UI update
        resBox.className = data.score > 0 
            ? "result-box risk-danger" 
            : "result-box risk-healthy";

        // Helper to determine bar colors
        let getScoreColor = (score, isQuantum) => {
            if (isQuantum) return score > 0.35 ? '#ff4d4d' : (score > -0.1 ? '#fbbf24' : '#00cfd5');
            return score < 0.3 ? '#ff4d4d' : (score < 0.6 ? '#fbbf24' : '#00cfd5');
        };

        let quantumPercent = Math.min(100, Math.max(0, (data.score + 1) * 50));
        let futurePercent = data.future_health !== null ? Math.min(100, Math.max(0, data.future_health * 100)) : 0;

        resBox.innerHTML = `
            <h2 class="animate-down" style="margin-bottom:5px;">${data.status}</h2>
            <p class="animate-fade" style="margin-bottom:20px; font-weight: 500; color: #a8b7c8;">Trend: <span style="color:white">${data.trend}</span></p>

            <div class="agent-box animate-up" style="background: rgba(0, 229, 255, 0.1); border-left: 4px solid ${getScoreColor(data.score, true)}; padding: 15px; border-radius: 8px; margin-bottom: 25px; text-align: left; animation-delay: 0.1s;">
                <small style="color: #00e5ff; font-weight: bold; text-transform: uppercase; letter-spacing: 1px;">🤖 Auto-Decision Agent</small>
                <p style="margin-top: 5px; font-size: 16px; color: white; line-height: 1.5;">${data.decision}</p>
            </div>

            <div class="ds-grid">
                <div class="ds-card ds-priority animate-up" style="animation-delay: 0.2s;">
                    <small>Maintenance Priority (Unit 2)</small>
                    <span>${data.priority}</span>
                </div>
                <div class="ds-card ds-spatial animate-up" style="animation-delay: 0.2s;">
                    <small>Nearest Hangar (Unit 5)</small>
                    <span>${data.hangar}</span>
                </div>
                <div class="ds-card ds-history animate-up" style="animation-delay: 0.3s;">
                    <small>Log Version (Unit 6)</small>
                    <span>v${data.history_version}</span>
                </div>
                <div class="ds-card ds-score animate-up" style="animation-delay: 0.4s;">
                    <small>Quantum Score</small>
                    <span>${data.score.toFixed(4)}</span>
                    <div class="progress-bar-bg"><div class="progress-bar-fill" style="width: ${quantumPercent}%; background: ${getScoreColor(data.score, true)}"></div></div>
                </div>
                <div class="ds-card ds-future animate-up" style="animation-delay: 0.5s;">
                    <small>Future LSTM Status</small>
                    <span style="color: ${data.future_status.includes('Failure') ? '#ff4d4d' : (data.future_status.includes('Moderate') ? '#fbbf24' : '#00cfd5')}">${data.future_status}</span>
                </div>
                <div class="ds-card ds-future-score animate-up" style="animation-delay: 0.6s;">
                    <small>Future Health Score</small>
                    <span>${(data.future_health !== null && data.future_health !== undefined) ? data.future_health.toFixed(4) : "N/A"}</span>
                    ${data.future_health !== null && data.future_health !== undefined ? 
                        `<div class="progress-bar-bg"><div class="progress-bar-fill" style="width: ${futurePercent}%; background: ${getScoreColor(data.future_health, false)}"></div></div>` : ''}
                </div>
                
                <div class="ds-card animate-up" style="animation-delay: 0.7s; border-left: 3px solid #ff4d4d; background: rgba(255, 77, 77, 0.05);">
                    <small>Fleet Most Critical (DEPQ)</small>
                    <span style="font-size: 16px; margin-top: 5px; color: white; display: block;">${data.depq_max_engine ? data.depq_max_engine : 'N/A'}</span>
                    <span style="font-size: 13px; color: #ff4d4d;">Score: ${data.depq_max_score !== null ? data.depq_max_score.toFixed(4) : '-'}</span>
                </div>
                <div class="ds-card animate-up" style="animation-delay: 0.8s; border-left: 3px solid #00cfd5; background: rgba(0, 207, 213, 0.05);">
                    <small>Fleet Healthiest (DEPQ)</small>
                    <span style="font-size: 16px; margin-top: 5px; color: white; display: block;">${data.depq_min_engine ? data.depq_min_engine : 'N/A'}</span>
                    <span style="font-size: 13px; color: #00cfd5;">Score: ${data.depq_min_score !== null ? data.depq_min_score.toFixed(4) : '-'}</span>
                </div>
            </div>
            <p class="animate-fade" style="font-size: 12px; margin-top: 15px; color: #94a3b8; animation-delay: 0.8s;">
                Analyzing from: ${lat.toFixed(4)}, ${lon.toFixed(4)}
            </p>
        `;

        // 🔥 ✅ CORRECT PLACE FOR MAP UPDATE
        updateMap(
            data.processed_at[0],   // aircraft lat
            data.processed_at[1],   // aircraft lon
            data.hangar_coords[0],  // hangar lat
            data.hangar_coords[1]   // hangar lon
        );
    })
    .catch(err => {
        console.error("Error:", err);
        document.getElementById("result").innerHTML = `<h2>Frontend Error: ${err.message}</h2><p>Please check the terminal for backend errors.</p>`;
    });
}