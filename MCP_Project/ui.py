# app.py
import os
import uuid
import asyncio
from pathlib import Path
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, session, render_template_string

# Your async agent entrypoint:
from client import run_agent

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key")

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
DOCS_DIR.mkdir(exist_ok=True)

# -------------- Helpers --------------

def get_thread_id() -> str:
    if "thread_id" not in session:
        session["thread_id"] = str(uuid.uuid4())
    return session["thread_id"]

def parse_files_prefix(user_text: str):
    """
    If input begins with 'file ' or 'files ', parse subsequent tokens as filenames.
    Returns (filenames, remainder_text) OR ([], original_text) if no prefix.
    """
    raw = (user_text or "").strip()
    lowered = raw.lower()
    if not (lowered.startswith("file ") or lowered.startswith("files ")):
        return [], raw

    tokens = raw.split()
    tokens = tokens[1:]  # drop 'file(s)'
    fnames, remainder_tokens = [], []
    for t in tokens:
        t_clean = os.path.basename(t.strip().strip(","))  # allow commas, strip paths
        if "." in t_clean and " " not in t_clean and t_clean not in (".", ".."):
            fnames.append(t_clean)
        else:
            remainder_tokens.append(t)
    # de-dupe, preserve order
    seen, out = set(), []
    for n in fnames:
        if n and n not in seen:
            seen.add(n); out.append(n)
    return out, " ".join(remainder_tokens).strip()

def build_name_preamble(file_names):
    """
    Build a one-time instruction that only lists file NAMES.
    No file contents are inlined.
    """
    if not file_names:
        return ""
    listed = ", ".join(file_names)
    return (
        "One-time setup:\n"
        f"- Please read the following documents from the ./docs directory: {listed}\n"
        "- Use them as context for this conversation. You only need to read them once now.\n"
        "- Acknowledge when you have read them."
    )

# -------------- Routes --------------

@app.route("/", methods=["GET"])
def index():
    _ = get_thread_id()
    return render_template_string(HTML_PAGE)

@app.route("/chat", methods=["POST"])
def chat():
    thread_id = get_thread_id()
    user_text = (request.form.get("message") or "").strip()

    # 1) Save uploads to ./docs and remember this batch in the session
    uploaded_files = request.files.getlist("files")
    saved_names = []
    for f in uploaded_files:
        if not f or not f.filename:
            continue
        safe_name = secure_filename(f.filename)
        target = DOCS_DIR / safe_name
        target.parent.mkdir(parents=True, exist_ok=True)
        f.save(target)
        saved_names.append(safe_name)

    # If there were uploads in THIS request, set a one-time pending preamble for them.
    if saved_names:
        session["pending_files"] = saved_names

    # 2) Parse an explicit files-prefix, if the user typed one
    prefix_files, remainder = parse_files_prefix(user_text)

    final_prompt = None

    # 3) Highest priority: one-time preamble for the most recent upload batch (if pending)
    pending_files = session.get("pending_files") or []
    if pending_files:
        preamble = build_name_preamble(pending_files)
        body_text = user_text if not prefix_files else (remainder or user_text)
        final_prompt = f"{preamble}\n\nUser request:\n{body_text or 'Give me a summary of these documents.'}".strip()
        # Clear so it only happens ONCE per upload event
        session["pending_files"] = []
    # 4) Otherwise, if user typed "file/files ..." in this message, send preamble ONCE for that set
    elif prefix_files:
        preamble = build_name_preamble(prefix_files)
        body_text = remainder or "Give me a summary of these documents."
        final_prompt = f"{preamble}\n\nUser request:\n{body_text}".strip()
    # 5) Otherwise, normal prompt
    else:
        final_prompt = user_text

    if not final_prompt:
        return jsonify({"ok": False, "error": "Empty message."}), 400

    # 6) Call your async agent
    try:
        response_text = asyncio.run(run_agent(final_prompt, thread_id=thread_id))
    except RuntimeError:
        # if already in an event loop
        response_text = asyncio.get_event_loop().run_until_complete(
            run_agent(final_prompt, thread_id=thread_id)
        )

    return jsonify({
        "ok": True,
        "user": user_text,
        "saved_files": saved_names,       # files uploaded in THIS message
        "prefix_files": prefix_files,     # files named via "file(s) ..." in THIS message
        "agent": response_text
    })

# ---------- Inline HTML+CSS+JS (no external files) ----------
HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Shystem</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <meta http-equiv="X-UA-Compatible" content="IE=edge" />
  <style>
    :root{
      --bg: #0b0f17;
      --bg2: #0f1117;
      --panel: #151922;
      --accent: #3b82f6;
      --text: #e5e7eb;
      --muted: #9ca3af;
      --user-bubble: #2563eb;
      --agent-bubble: #232938;
      --shadow: rgba(0,0,0,0.35);
      --ring: rgba(59,130,246,0.35);
    }
    *{ box-sizing: border-box; }
    html, body{
      height: 100%;
      margin: 0;
      -webkit-font-smoothing: antialiased;
      -moz-osx-font-smoothing: grayscale;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, "Helvetica Neue", Arial, "Apple Color Emoji","Segoe UI Emoji";
      color: var(--text);
      background:
        radial-gradient(1200px 800px at 50% -10%, #1a2030 0%, var(--bg2) 60%, var(--bg) 100%),
        linear-gradient(180deg, rgba(59,130,246,0.06), transparent 30%);
      overflow: hidden; /* prevent double scrollbars */
    }
    .app{
      display: grid;
      grid-template-rows: auto 1fr auto;
      height: 100vh;   /* ensure full viewport height */
      max-width: 980px;
      margin: 0 auto;
      padding: 0 14px;
    }
    .app-header{
      position: sticky;
      top: 0;
      z-index: 10;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 18px 0 10px;
      background: linear-gradient(180deg, rgba(11,15,23,0.85), rgba(11,15,23,0.35) 70%, transparent);
      backdrop-filter: blur(6px);
    }
    .title{
      font-weight: 900;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: #eaf0ff;
      text-align: center;
      font-size: clamp(20px, 2.6vw, 28px);
      text-shadow: 0 10px 30px rgba(59,130,246,0.35);
    }
    .chat-window{
      position: relative;
      overflow-y: auto;
      padding: 18px 0 8px;
      scrollbar-gutter: stable both-edges;
    }
    .chat-window::-webkit-scrollbar{
      width: 10px;
      background: transparent;
    }
    .chat-window::-webkit-scrollbar-thumb{
      background: rgba(255,255,255,0.08);
      border-radius: 10px;
    }
    .stream{
      display: flex;
      flex-direction: column;
      gap: 10px;
      padding-bottom: 8px;
    }
    .welcome{
      margin: 2px auto 14px;
      max-width: 620px;
      width: 95%;
      padding: 14px 16px;
      background: rgba(255,255,255,0.04);
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 14px;
      color: var(--muted);
      text-align: center;
      box-shadow: 0 20px 50px var(--shadow);
    }
    .msg{ display: flex; }
    .msg.user{ justify-content: flex-end; }
    .msg.agent{ justify-content: flex-start; }
    .bubble{
      max-width: min(78%, 720px);
      padding: 12px 14px;
      border-radius: 16px;
      line-height: 1.48;
      box-shadow: 0 10px 28px var(--shadow);
      white-space: pre-wrap;
      word-break: break-word;
      border: 1px solid rgba(255,255,255,0.06);
      transform: translateZ(0);
    }
    .msg.user .bubble{
      background: linear-gradient(180deg, var(--user-bubble), #1d4ed8);
      color: #eef3ff;
      border-top-right-radius: 8px;
      outline: 2px solid transparent;
    }
    .msg.agent .bubble{
      background: linear-gradient(180deg, #1d2230, var(--agent-bubble));
      color: #dbe3ff;
      border-top-left-radius: 8px;
    }
    .files-tag{
      background: #334155 !important;
      color: #e5e7eb !important;
    }

    .composer-wrap{
      position: sticky;
      bottom: 0;
      z-index: 15;
      background: linear-gradient(180deg, transparent, rgba(11,15,23,0.7) 25%, rgba(11,15,23,0.95));
      padding: 14px 0 18px;
    }
    .composer{
      display: grid;
      grid-template-columns: auto 1fr auto;
      gap: 10px;
      align-items: center;
    }
    .file-label{
      display: grid;
      place-items: center;
      width: 44px;
      height: 44px;
      border-radius: 14px;
      background: var(--panel);
      box-shadow: 0 8px 28px var(--shadow);
      cursor: pointer;
      transition: transform .05s ease, box-shadow .2s;
      border: 1px solid rgba(255,255,255,0.06);
    }
    .file-label:hover{ transform: translateY(-1px); box-shadow: 0 12px 32px var(--shadow); }
    .file-label input{ display: none; }
    .file-label span{ font-size: 18px; }
    #message{
      width: 100%;
      border: 1px solid rgba(255,255,255,0.1);
      background: rgba(21,25,34,0.9);
      border-radius: 14px;
      color: var(--text);
      padding: 12px 14px;
      outline: none;
      box-shadow: 0 0 0 0 var(--ring), inset 0 0 0 1px rgba(255,255,255,0.03);
      transition: box-shadow 0.15s ease, border-color 0.15s ease;
    }
    #message:focus{
      border-color: rgba(59,130,246,0.45);
      box-shadow: 0 0 0 4px var(--ring), inset 0 0 0 1px rgba(59,130,246,0.25);
    }
    #message::placeholder{ color: #97a2b3; }
    #send{
      width: 48px;
      height: 44px;
      border-radius: 14px;
      border: none;
      cursor: pointer;
      background: var(--accent);
      color: white;
      font-size: 18px;
      box-shadow: 0 10px 28px var(--shadow);
      transition: transform .05s ease, filter .15s;
    }
    #send:hover{ transform: translateY(-1px); filter: brightness(1.08); }

    @media (max-width: 520px){
      .bubble{ max-width: 90%; }
      .title{ letter-spacing: 0.08em; }
    }
  </style>
</head>
<body>
  <div class="app" role="application">
    <header class="app-header">
      <div class="title" aria-label="Brand">Shystem</div>
    </header>

    <main id="chat" class="chat-window" aria-live="polite" aria-label="Chat messages">
      <div class="stream" id="stream">
        <div class="welcome">
          <div>🤖 MCP + LangGraph Agent is ready.</div>
          <div style="margin-top:6px"><b>Tip:</b> Upload files or start with <code>file</code>/<code>files</code> and filenames from <code>/docs</code>.</div>
        </div>
      </div>
    </main>

    <div class="composer-wrap">
      <form id="composer" class="composer" enctype="multipart/form-data" autocomplete="off">
        <label class="file-label" title="Attach files">
          <input id="file-input" type="file" name="files" multiple />
          <span>📎</span>
        </label>
        <input id="message" name="message" type="text" placeholder="Type a message… (e.g., files report.txt notes.md Summarize)" required />
        <button id="send" type="submit" title="Send">➤</button>
      </form>
    </div>
  </div>

  <script>
    const stream = document.getElementById("stream");
    const form = document.getElementById("composer");
    const messageInput = document.getElementById("message");
    const fileInput = document.getElementById("file-input");
    const chatWindow = document.getElementById("chat");

    function scrollToBottom(){
      chatWindow.scrollTop = chatWindow.scrollHeight;
    }
    function bubble(text, who = "user"){
      const wrap = document.createElement("div");
      wrap.className = `msg ${who}`;
      const b = document.createElement("div");
      b.className = "bubble";
      b.textContent = text;
      wrap.appendChild(b);
      stream.appendChild(wrap);
      scrollToBottom();
    }
    function filesTag(names){
      if (!names || !names.length) return;
      const wrap = document.createElement("div");
      wrap.className = "msg user";
      const b = document.createElement("div");
      b.className = "bubble files-tag";
      b.textContent = `Uploaded to /docs: ${names.join(", ")}`;
      wrap.appendChild(b);
      stream.appendChild(wrap);
      scrollToBottom();
    }

    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const text = messageInput.value.trim();
      const files = fileInput.files;
      if (!text && (!files || !files.length)) return;

      if (text) bubble(text, "user");

      const data = new FormData();
      data.append("message", text);
      if (files && files.length){
        for (let i = 0; i < files.length; i++) data.append("files", files[i]);
      }

      // Clear inputs optimistically
      messageInput.value = "";
      fileInput.value = "";

      try{
        const r = await fetch("/chat", { method: "POST", body: data });
        const json = await r.json();

        if (!json.ok){
          bubble(`Error: ${json.error || "Something went wrong."}`, "agent");
          return;
        }
        if (json.saved_files?.length) filesTag(json.saved_files);
        if (json.prefix_files?.length) bubble(`Reading from /docs: ${json.prefix_files.join(", ")}`, "agent");

        bubble(json.agent, "agent");
      }catch(err){
        bubble(`Network error: ${err}`, "agent");
      }
    });

    window.addEventListener("load", () => {
      messageInput.focus();
      scrollToBottom();
    });
  </script>
</body>
</html>
"""

if __name__ == "__main__":
    # pip install flask werkzeug
    # Run: python app.py  -> open http://127.0.0.1:5000
    app.run(debug=True)