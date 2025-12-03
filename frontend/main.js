const COLUMNS = "ABCDEFGHIJK".split("");
const ROWS = Array.from({ length: 18 }, (_, i) => i + 1);

const gridEl = document.getElementById("grid");
const holdsCountEl = document.getElementById("holdsCount");
const gradeValueEl = document.getElementById("gradeValue");
const sequenceBoxEl = document.getElementById("sequenceBox");
const resultEl = document.getElementById("result");
const predictBtn = document.getElementById("predictBtn");
const clearBtn = document.getElementById("clearBtn");

const selectedOrder = [];
const selectedSet = new Set();
const cellMap = new Map();

for (let r = ROWS.length - 1; r >= 0; r--) {
  const rowNum = ROWS[r];

  for (let c = 0; c < COLUMNS.length; c++) {
    const col = COLUMNS[c];
    const holdId = `${col}${rowNum}`;

    const cell = document.createElement("div");
    cell.className = "cell";
    cell.dataset.holdId = holdId;

    const dot = document.createElement("div");
    dot.className = "cell-dot";
    cell.appendChild(dot);

    cell.addEventListener("click", () => toggleHold(holdId));

    gridEl.appendChild(cell);
    cellMap.set(holdId, cell);
  }
}

function toggleHold(holdId) {
  if (selectedSet.has(holdId)) {
    selectedSet.delete(holdId);
    const idx = selectedOrder.indexOf(holdId);
    if (idx !== -1) selectedOrder.splice(idx, 1);
    cellMap.get(holdId)?.classList.remove("selected");
  } else {
    selectedSet.add(holdId);
    selectedOrder.push(holdId);
    cellMap.get(holdId)?.classList.add("selected");
  }

  updateUI();
}

function updateUI() {
  holdsCountEl.textContent = String(selectedOrder.length);

  if (selectedOrder.length === 0) {
    sequenceBoxEl.textContent = "No holds selected.";
    predictBtn.disabled = true;
  } else {
    sequenceBoxEl.textContent = selectedOrder.join(" → ");
    predictBtn.disabled = false;
  }

  resultEl.textContent = "";
  resultEl.className = "";
}

clearBtn.addEventListener("click", () => {
  selectedSet.clear();
  selectedOrder.length = 0;

  for (const cell of cellMap.values()) {
    cell.classList.remove("selected");
  }

  gradeValueEl.textContent = "—";
  updateUI();
});

predictBtn.addEventListener("click", async () => {
  if (!selectedOrder.length) return;

  resultEl.textContent = "Predicting…";
  resultEl.className = "";

  try {
    const resp = await fetch("http://127.0.0.1:8000/v1/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ moves: selectedOrder }),
    });

    if (!resp.ok) {
      throw new Error(`HTTP ${resp.status}`);
    }

    const data = await resp.json();
    gradeValueEl.textContent = data.grade || "—";
    resultEl.textContent = `Model sequence (sorted): ${
      (data.sorted_moves || []).join(" ")
    }`;
    resultEl.className = "result-ok";
  } catch (err) {
    console.error(err);
    resultEl.textContent = "Error calling backend.";
    resultEl.className = "result-error";
  }
});

updateUI();
