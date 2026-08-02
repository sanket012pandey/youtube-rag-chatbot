const API_URL = "http://127.0.0.1:8000/ask";

let currentVideoId = null;

const statusEl = document.getElementById("video-status");
const chatEl = document.getElementById("chat");
const questionEl = document.getElementById("question");
const askBtn = document.getElementById("ask-btn");

// Step 1: Detect the current tab's YouTube video ID
chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
  const url = tabs[0]?.url || "";
  const match = url.match(/[?&]v=([a-zA-Z0-9_-]{11})/);

  if (match) {
    currentVideoId = match[1];
    statusEl.textContent = `Connected to video: ${currentVideoId}`;
  } else {
    statusEl.textContent = "Not a YouTube video page. Open a video first.";
    statusEl.classList.add("error");
    askBtn.disabled = true;
  }
});

// Step 2: Add a message bubble to the chat area
function addMessage(text, type) {
  const div = document.createElement("div");
  div.className = `msg ${type}`;
  div.textContent = text;
  chatEl.appendChild(div);
  chatEl.scrollTop = chatEl.scrollHeight;
  return div;
}

// Step 3: Ask the API
async function askQuestion() {
  const question = questionEl.value.trim();
  if (!question || !currentVideoId) return;

  addMessage(question, "user");
  questionEl.value = "";
  askBtn.disabled = true;

  const loadingMsg = addMessage("Thinking...", "loading");

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        video_id: currentVideoId,
        question: question
      })
    });

    loadingMsg.remove();

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      addMessage(`Error: ${err.detail || response.statusText}`, "bot");
    } else {
      const data = await response.json();
      addMessage(data.answer, "bot");
    }
  } catch (e) {
    loadingMsg.remove();
    addMessage("Could not reach the API. Is your Docker container running on port 8000?", "bot");
  } finally {
    askBtn.disabled = false;
  }
}

askBtn.addEventListener("click", askQuestion);
questionEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter") askQuestion();
});