"""
app.py — Multilingual AI Communication System
=============================================
Identical to your original app.py EXCEPT:
  • formalize_with_gemini()  → model_router.formalize_with_local_models()
  • transcribe()             → model_router.transcribe_audio_local()
  • explain_changes()        → model_router.explain_changes_local()
  • /api/router_status       → new health endpoint
  • Gemini is optional fallback if GEMINI_API_KEY present

Environment variables (set in .env):
  ASR_MODEL_PATH   = ./saved_models/whisper_indic_asr_merged
  FORM_MODEL_PATH  = ./saved_models/formalization_model_merged
  TRANS_MODEL_PATH = ./saved_models/translation_model_merged
  GEMINI_API_KEY   = (optional — for fallback)
  FLASK_SECRET_KEY = <random>
  DATABASE_PATH    = text_formalizer.db
"""

import os, io, json, re, subprocess, wave, tempfile, secrets, sqlite3
from datetime import datetime
from flask import (
    Flask, render_template, request, jsonify, session,
    redirect, url_for, send_file, Response, stream_with_context,
)
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import PyPDF2
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_JUSTIFY
from authlib.integrations.flask_client import OAuth

# ── Load env ──────────────────────────────────────────────────
load_dotenv()

# ── Flask setup ───────────────────────────────────────────────
app = Flask(__name__)
app.secret_key              = os.getenv("FLASK_SECRET_KEY", secrets.token_hex(16))
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.config["PERMANENT_SESSION_LIFETIME"] = 86400

DATABASE_PATH = os.getenv("DATABASE_PATH", "text_formalizer.db")

# ── OAuth ─────────────────────────────────────────────────────
oauth = OAuth(app)
google = oauth.register(
    name="google",
    client_id    = os.getenv("GOOGLE_CLIENT_ID",    "placeholder"),
    client_secret= os.getenv("GOOGLE_CLIENT_SECRET","placeholder"),
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)
microsoft = oauth.register(
    name="microsoft",
    client_id    = os.getenv("MICROSOFT_CLIENT_ID",    "placeholder"),
    client_secret= os.getenv("MICROSOFT_CLIENT_SECRET","placeholder"),
    server_metadata_url=(
        "https://login.microsoftonline.com/common/v2.0"
        "/.well-known/openid-configuration"
    ),
    client_kwargs={"scope": "openid email profile User.Read"},
)

# ── Local model router ────────────────────────────────────────
try:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "model_router"))
    from router import (
        formalize_with_local_models,
        transcribe_audio_local,
        explain_changes_local,
        router_status,
    )
    USE_LOCAL_MODELS = True
    print("✅ Local model router loaded")
except ImportError as e:
    USE_LOCAL_MODELS = False
    print(f"⚠️  Local router not found ({e}), using Gemini fallback")

# ── Optional Gemini fallback ──────────────────────────────────
gemini_model = None
if not USE_LOCAL_MODELS or os.getenv("FORCE_GEMINI"):
    try:
        import google.generativeai as genai
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)
            gemini_model = genai.GenerativeModel("gemini-2.5-flash")
            print("✅ Gemini fallback loaded")
    except ImportError:
        pass

# ── Static maps ───────────────────────────────────────────────
LANGUAGE_CODES = {
    "english":"en-IN","hindi":"hi-IN","telugu":"te-IN","marathi":"mr-IN",
    "tamil":"ta-IN","bengali":"bn-IN","gujarati":"gu-IN","malayalam":"ml-IN",
    "kannada":"kn-IN","spanish":"es-ES","french":"fr-FR","german":"de-DE",
    "japanese":"ja-JP","mandarin":"zh-CN","hinglish":"hi-Latn-IN",
}
FORMALITY_LEVELS = {
    "casual":"conversational and friendly",
    "professional":"professional and polite",
    "academic":"academic and scholarly",
}
CONTEXT_FORMATS = {
    "email"       :{"name":"Email","description":"Professional email","structure":"Format as a professional email with appropriate greeting, body paragraphs, and closing signature."},
    "report"      :{"name":"Report","description":"Structured report","structure":"Format as a structured report with clear sections and professional language."},
    "meeting_notes":{"name":"Meeting Notes","description":"Organized notes","structure":"Format as organized meeting notes with key points and action items."},
    "presentation":{"name":"Presentation","description":"Slide-style bullets","structure":"Format as presentation content with clear bullet points and concise statements."},
    "proposal"    :{"name":"Proposal","description":"Formal proposal","structure":"Format as a formal proposal with clear objectives, methodology, and professional tone."},
    "general"     :{"name":"General","description":"Standard formal","structure":"Convert to formal language without specific formatting."},
    "legal"       :{"name":"Legal","description":"Legal document","structure":"Format using precise legal language with formal document structure."},
    "academic"    :{"name":"Academic","description":"Academic writing","structure":"Format using scholarly tone, passive voice where appropriate, and academic conventions."},
}

# ── Database ──────────────────────────────────────────────────
def init_db():
    conn   = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            original_text TEXT NOT NULL,
            formalized_text TEXT NOT NULL,
            selected_language TEXT NOT NULL,
            detected_language TEXT,
            output_mode TEXT NOT NULL,
            formality_level TEXT DEFAULT 'professional',
            context_format TEXT DEFAULT 'general',
            custom_formality_score INTEGER,
            original_formality_score REAL,
            formalized_formality_score REAL,
            word_count INTEGER,
            improvement_score REAL,
            session_id TEXT,
            user_id INTEGER,
            model_backend TEXT DEFAULT 'local'
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            provider TEXT NOT NULL,
            provider_id TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    for col in ["user_id INTEGER", "model_backend TEXT DEFAULT 'local'"]:
        try:
            cursor.execute(f"ALTER TABLE conversions ADD COLUMN {col}")
        except sqlite3.OperationalError:
            pass
    conn.commit()
    conn.close()
    print("✅ Database ready")

init_db()

# ── Helpers (unchanged from original) ────────────────────────
def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg","-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def convert_audio_to_wav_ffmpeg(input_path, output_path):
    try:
        subprocess.run(
            ["ffmpeg","-i",input_path,"-acodec","pcm_s16le","-ar","16000","-ac","1","-y",output_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
        )
        return True
    except Exception:
        return False

def save_conversion_to_db(data, session_id, backend="local"):
    try:
        conn   = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO conversions
            (original_text,formalized_text,selected_language,detected_language,
             output_mode,formality_level,context_format,custom_formality_score,
             original_formality_score,formalized_formality_score,
             word_count,improvement_score,session_id,user_id,model_backend)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            data["original_text"], data["formalized_text"],
            data["selected_language"], data.get("detected_language",""),
            data["output_mode"], data.get("formality_level","professional"),
            data.get("context","general"), data.get("custom_formality_score"),
            data.get("analytics",{}).get("original",{}).get("formality_score",0),
            data.get("analytics",{}).get("formalized",{}).get("formality_score",0),
            data.get("analytics",{}).get("original",{}).get("word_count",0),
            data.get("analytics",{}).get("improvement",0),
            session_id, session.get("user_id"), backend,
        ))
        conn.commit(); conn.close()
        return True
    except Exception as e:
        print(f"DB error: {e}"); return False

def get_conversions_from_db(limit=50):
    try:
        conn   = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        if "user_id" in session:
            cursor.execute("""
                SELECT id,timestamp,original_text,formalized_text,selected_language,
                       output_mode,formality_level,context_format,custom_formality_score,
                       original_formality_score,formalized_formality_score,improvement_score
                FROM conversions WHERE user_id=? OR session_id=?
                ORDER BY timestamp DESC LIMIT ?
            """, (session["user_id"], session.get("session_id",""), limit))
        else:
            cursor.execute("""
                SELECT id,timestamp,original_text,formalized_text,selected_language,
                       output_mode,formality_level,context_format,custom_formality_score,
                       original_formality_score,formalized_formality_score,improvement_score
                FROM conversions WHERE session_id=?
                ORDER BY timestamp DESC LIMIT ?
            """, (session.get("session_id",""), limit))
        rows = cursor.fetchall(); conn.close()
        return [{
            "id":r[0],"timestamp":r[1],
            "original_text":r[2][:100]+"..." if len(r[2])>100 else r[2],
            "formalized_text":r[3][:100]+"..." if len(r[3])>100 else r[3],
            "full_original":r[2],"full_formalized":r[3],
            "language":r[4],"output_mode":r[5],"formality_level":r[6],
            "context":r[7],"custom_formality_score":r[8],
            "analytics":{"original":{"formality_score":r[9]},
                         "formalized":{"formality_score":r[10]},
                         "improvement":r[11]},
        } for r in rows]
    except Exception as e:
        print(f"DB error: {e}"); return []

def init_session():
    if "history"    not in session: session["history"]    = []
    if "settings"   not in session: session["settings"]   = {"theme":"light","default_language":"english","default_formality":"professional","show_analytics":True}
    if "session_id" not in session: session["session_id"] = secrets.token_hex(16)

def detect_language(text):
    checks = [
        ('\u0900','\u097F','Hindi'), ('\u0C00','\u0C7F','Telugu'),
        ('\u0B80','\u0BFF','Tamil'), ('\u0980','\u09FF','Bengali'),
        ('\u0A80','\u0AFF','Gujarati'), ('\u0D00','\u0D7F','Malayalam'),
        ('\u0C80','\u0CFF','Kannada'),
        ('\u3040','\u309F','Japanese'), ('\u4E00','\u9FFF','Mandarin'),
    ]
    total = max(len(text), 1)
    for lo, hi, name in checks:
        if sum(1 for c in text if lo<=c<=hi) / total > 0.3:
            return name
    return "English"

def analyze_text(text, target_formality=None):
    words     = text.split()
    sentences = [s for s in text.split(".") if s.strip()]
    formal    = ["therefore","furthermore","consequently","accordingly","hence","thus","nevertheless","moreover"]
    informal  = ["gonna","wanna","yeah","stuff","kinda","sorta"]
    fc = sum(1 for w in words if w.lower() in formal)
    ic = sum(1 for w in words if w.lower() in informal)
    score = target_formality if target_formality is not None else min(100, max(0, 50+(fc-ic)*10))
    return {"word_count":len(words),"sentence_count":len(sentences),"formality_score":round(score,1)}

def save_to_history(result):
    if "history" not in session: session["history"] = []
    item = result.copy(); item["id"] = len(session["history"])+1; item["timestamp"] = datetime.now().isoformat()
    session["history"].insert(0, item)
    if len(session["history"]) > 50: session["history"] = session["history"][:50]
    session.modified = True

def extract_text_from_pdf(pdf_file):
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        return "\n".join(p.extract_text() for p in reader.pages).strip()
    except Exception as e:
        print(f"PDF error: {e}"); return ""

def create_formalized_pdf(original_text, formalized_text, language, formality_level):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    doc = SimpleDocTemplate(tmp.name, pagesize=letter)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("CustomTitle",parent=styles["Heading1"],fontSize=16,spaceAfter=12)
    body_style  = ParagraphStyle("CustomBody",parent=styles["BodyText"],fontSize=11,alignment=TA_JUSTIFY,spaceAfter=12)
    story = [
        Paragraph("Formalized Document", title_style),
        Spacer(1, 0.2*inch),
        Paragraph(f"<b>Language:</b> {language} | <b>Formality:</b> {formality_level.title()} | <b>Date:</b> {datetime.now():%B %d, %Y}", styles["Normal"]),
        Spacer(1, 0.3*inch),
        Paragraph("<b>Original Text:</b>", styles["Heading2"]),
        Spacer(1, 0.1*inch),
        Paragraph(original_text.replace("\n","<br/>"), body_style),
        Spacer(1, 0.3*inch),
        Paragraph("<b>Formalized Text:</b>", styles["Heading2"]),
        Spacer(1, 0.1*inch),
        Paragraph(formalized_text.replace("\n","<br/>"), body_style),
    ]
    doc.build(story)
    return tmp.name

# ── Core formalization dispatcher ─────────────────────────────
def run_formalization(text, language, output_mode, formality_level,
                      custom_formality, context, stream=False):
    """
    Routes to local models or Gemini fallback.
    Always returns the same interface as the original formalize_with_gemini().
    """
    if USE_LOCAL_MODELS:
        return formalize_with_local_models(
            text, language, output_mode,
            formality_level, custom_formality, context, stream=stream
        )
    elif gemini_model:
        return _gemini_formalize(text, language, output_mode,
                                 formality_level, custom_formality, context, stream)
    else:
        msg = "No inference backend available. Train a model or provide GEMINI_API_KEY."
        return (_chunk_str(msg) if stream else msg)

def _chunk_str(s, n=30):
    words = s.split()
    for i in range(0, len(words), n):
        yield " ".join(words[i:i+n]) + " "

def _gemini_formalize(text, language, output_mode, formality_level,
                      custom_formality, context, stream):
    """Original Gemini path — kept as fallback."""
    ctx_info = CONTEXT_FORMATS.get(context, CONTEXT_FORMATS["general"])
    if custom_formality is not None:
        f_instr = f"The formality level should be {custom_formality}/100. "
    else:
        f_instr = f"Make the text {FORMALITY_LEVELS.get(formality_level,'professional')}. "
    if output_mode == "same_language":
        prompt = (f"You are a professional text formalizer.\n"
                  f"Task: Convert the following {language} text to formal {language}.\n"
                  f"{f_instr}{ctx_info['structure']}\nText:\n{text}\n"
                  f"Provide ONLY the formalized text, no explanations.")
    else:
        prompt = (f"You are a professional translator and text formalizer.\n"
                  f"Task: Translate this {language} text to formal English.\n"
                  f"{f_instr}{ctx_info['structure']}\nText:\n{text}\n"
                  f"Provide ONLY the formalized English translation.")
    if stream:
        resp = gemini_model.generate_content(prompt, stream=True)
        def gen():
            for chunk in resp:
                if chunk.text:
                    yield chunk.text.replace("**","")
        return gen()
    else:
        resp = gemini_model.generate_content(prompt)
        out  = resp.text.strip().replace("**","")
        out  = re.sub(r"^\s*\*\s+","- ", out, flags=re.MULTILINE)
        return out

# ── Routes ────────────────────────────────────────────────────
@app.route("/")
@app.route("/home")
def home():
    init_session()
    db_history = get_conversions_from_db()
    stats = {"total_conversions":len(db_history),"languages_supported":len(LANGUAGE_CODES),"avg_improvement":0}
    if db_history:
        imps = [i["analytics"].get("improvement",0) for i in db_history]
        if imps: stats["avg_improvement"] = round(sum(imps)/len(imps),1)
    return render_template("home.html", stats=stats, settings=session.get("settings",{}))

@app.route("/converter")
def converter():
    init_session()
    return render_template("converter.html", settings=session.get("settings",{}))

@app.route("/history")
def history():
    init_session()
    return render_template("history.html", history=get_conversions_from_db(), settings=session.get("settings",{}))

@app.route("/settings")
def settings():
    init_session()
    return render_template("settings.html", settings=session.get("settings",{}))

@app.route("/login")
def login_page():
    init_session()
    return render_template("login.html", settings=session.get("settings",{}))

@app.route("/login/<provider>")
def login(provider):
    redirect_uri = url_for("auth", provider=provider, _external=True)
    if provider == "google":    return google.authorize_redirect(redirect_uri)
    if provider == "microsoft": return microsoft.authorize_redirect(redirect_uri)
    return redirect(url_for("home"))

@app.route("/auth/<provider>")
def auth(provider):
    try:
        if provider == "google":
            token     = google.authorize_access_token()
            user_info = token.get("userinfo") or google.get("https://openidconnect.googleapis.com/v1/userinfo").json()
            email, name, provider_id = user_info["email"], user_info.get("name",""), user_info["sub"]
        elif provider == "microsoft":
            token     = microsoft.authorize_access_token()
            user_info = token.get("userinfo")
            email     = user_info.get("email") or user_info.get("preferred_username")
            name      = user_info.get("name","")
            provider_id = user_info.get("oid") or user_info.get("sub")
        else:
            return redirect(url_for("home"))

        conn = sqlite3.connect(DATABASE_PATH); cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE email=?", (email,))
        row = cursor.fetchone()
        if row:
            user_id = row[0]
            cursor.execute("UPDATE users SET name=?,provider=?,provider_id=? WHERE id=?",(name,provider,provider_id,user_id))
        else:
            cursor.execute("INSERT INTO users (email,name,provider,provider_id) VALUES (?,?,?,?)",(email,name,provider,provider_id))
            user_id = cursor.lastrowid
        guest_sid = session.get("session_id")
        if guest_sid:
            cursor.execute("UPDATE conversions SET user_id=? WHERE session_id=? AND user_id IS NULL",(user_id,guest_sid))
        conn.commit(); conn.close()
        session["user_id"] = user_id; session["user_name"] = name; session["user_email"] = email
        return redirect(url_for("home"))
    except Exception as e:
        return f"Authentication failed: {e}", 400

@app.route("/logout")
def logout():
    session.pop("user_id",None); session.pop("user_name",None); session.pop("user_email",None)
    return redirect(url_for("home"))

# ── Main formalization API ────────────────────────────────────
@app.route("/api/formalize", methods=["POST"])
def formalize():
    try:
        init_session()
        data             = request.json
        input_text       = data.get("text","").strip()
        selected_language= data.get("language","english").lower()
        output_mode      = data.get("output_mode","same_language")
        formality_level  = data.get("formality_level","professional")
        custom_formality = data.get("custom_formality")
        context          = data.get("context","general")

        if not input_text:
            return jsonify({"error":"Please provide text to formalize"}), 400

        detected_language = detect_language(input_text)
        original_analytics= analyze_text(input_text)
        backend_used      = "local" if USE_LOCAL_MODELS else "gemini"

        def generate():
            try:
                gen = run_formalization(
                    input_text, selected_language.title(),
                    output_mode, formality_level, custom_formality, context, stream=True
                )
                formalized_full = ""
                for chunk in gen:
                    formalized_full += chunk
                    yield f"data: {json.dumps({'chunk':chunk})}\n\n"

                tf = custom_formality if custom_formality is not None else None
                formalized_analytics = analyze_text(formalized_full, tf)
                formalized_full = re.sub(r"^\s*\*\s+","- ",formalized_full,flags=re.MULTILINE).strip()

                result = {
                    "success"           : True,
                    "original_text"     : input_text,
                    "formalized_text"   : formalized_full,
                    "selected_language" : selected_language.title(),
                    "detected_language" : detected_language.title(),
                    "output_mode"       : "Same Language" if output_mode=="same_language" else "Formal English",
                    "formality_level"   : formality_level,
                    "context"           : context,
                    "context_name"      : CONTEXT_FORMATS.get(context,CONTEXT_FORMATS["general"])["name"],
                    "custom_formality_score": custom_formality,
                    "model_backend"     : backend_used,
                    "analytics": {
                        "original"  : original_analytics,
                        "formalized": formalized_analytics,
                        "improvement": round(
                            formalized_analytics["formality_score"]
                            - original_analytics["formality_score"], 1
                        ),
                    },
                }
                save_to_history(result)
                save_conversion_to_db(result, session.get("session_id"), backend_used)
                yield f"data: {json.dumps({'done':True,'result':result})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error':str(e)})}\n\n"

        return Response(stream_with_context(generate()), mimetype="text/event-stream")
    except Exception as e:
        return jsonify({"error":str(e)}), 500

# ── PDF upload / download ─────────────────────────────────────
@app.route("/api/upload-pdf", methods=["POST"])
def upload_pdf():
    try:
        if "pdf" not in request.files: return jsonify({"error":"No PDF provided"}), 400
        f = request.files["pdf"]
        if not f.filename.lower().endswith(".pdf"): return jsonify({"error":"File must be PDF"}), 400
        text = extract_text_from_pdf(f)
        if not text: return jsonify({"error":"Could not extract text"}), 400
        return jsonify({"success":True,"text":text})
    except Exception as e:
        return jsonify({"error":str(e)}), 500

@app.route("/api/download-pdf", methods=["POST"])
def download_pdf():
    try:
        data = request.json
        if not data.get("formalized_text"): return jsonify({"error":"No formalized text"}), 400
        path = create_formalized_pdf(data.get("original_text",""), data["formalized_text"],
                                     data.get("language","English"), data.get("formality_level","professional"))
        return send_file(path, mimetype="application/pdf", as_attachment=True,
                         download_name=f"formalized_{datetime.now():%Y%m%d_%H%M%S}.pdf")
    except Exception as e:
        return jsonify({"error":str(e)}), 500

# ── Transcription ─────────────────────────────────────────────
@app.route("/api/transcribe", methods=["POST"])
def transcribe():
    temp_input = temp_wav = None
    try:
        if "audio" not in request.files: return jsonify({"error":"No audio file"}), 400
        audio_file = request.files["audio"]
        language   = request.form.get("language","english").lower()

        if not check_ffmpeg():
            return jsonify({"error":"FFmpeg not installed. Required for audio processing."}), 500

        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".webm")
        temp_wav   = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        audio_file.save(temp_input.name); temp_input.close(); temp_wav.close()

        if not convert_audio_to_wav_ffmpeg(temp_input.name, temp_wav.name):
            return jsonify({"error":"Audio conversion failed"}), 400

        if USE_LOCAL_MODELS:
            # Use local Whisper model
            try:
                text = transcribe_audio_local(temp_wav.name, language)
                return jsonify({"success":True,"text":text,"language":language,"backend":"local_whisper"})
            except RuntimeError as e:
                print(f"Local ASR failed, falling back to Google SR: {e}")

        # Fallback: Google Speech Recognition
        import speech_recognition as sr
        recognizer = sr.Recognizer()
        lang_code  = LANGUAGE_CODES.get(language, "en-IN")
        with sr.AudioFile(temp_wav.name) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data, language=lang_code)
        return jsonify({"success":True,"text":text,"language":language,"backend":"google_sr"})

    except Exception as e:
        return jsonify({"error":f"Transcription failed: {e}"}), 500
    finally:
        for f in [temp_input, temp_wav]:
            try:
                if f and os.path.exists(f.name): os.unlink(f.name)
            except Exception: pass

# ── Explain changes ───────────────────────────────────────────
@app.route("/api/explain", methods=["POST"])
def explain_changes():
    try:
        init_session()
        data           = request.json
        original_text  = data.get("original_text","")
        formalized_text= data.get("formalized_text","")
        if not original_text or not formalized_text:
            return jsonify({"error":"Both original and formalized text required"}), 400

        if USE_LOCAL_MODELS:
            explanation = explain_changes_local(original_text, formalized_text)
        elif gemini_model:
            prompt = (f"Explain in 4-5 bullet points (plain text, no markdown) "
                      f"the key changes from:\nOriginal: {original_text}\n"
                      f"Formalized: {formalized_text}")
            resp = gemini_model.generate_content(prompt)
            explanation = resp.text.strip().replace("**","")
            explanation = re.sub(r"^\s*\*\s+","- ",explanation,flags=re.MULTILINE)
        else:
            explanation = "No backend available for explanation."

        return jsonify({"success":True,"explanation":explanation.strip()})
    except Exception as e:
        return jsonify({"error":str(e)}), 500

# ── Health / status ───────────────────────────────────────────
@app.route("/api/router_status")
def api_router_status():
    status = {
        "use_local_models": USE_LOCAL_MODELS,
        "gemini_available": gemini_model is not None,
    }
    if USE_LOCAL_MODELS:
        status.update(router_status())
    return jsonify(status)

# ── History management ────────────────────────────────────────
@app.route("/api/history/clear", methods=["POST"])
def clear_history():
    init_session(); session["history"] = []; session.modified = True
    try:
        conn = sqlite3.connect(DATABASE_PATH); cursor = conn.cursor()
        cursor.execute("DELETE FROM conversions WHERE session_id=?",(session.get("session_id"),))
        conn.commit(); conn.close()
    except Exception: pass
    return jsonify({"success":True})

@app.route("/api/history/delete/<int:item_id>", methods=["DELETE"])
def delete_history_item(item_id):
    init_session()
    session["history"] = [i for i in session.get("history",[]) if i["id"]!=item_id]
    session.modified = True
    try:
        conn = sqlite3.connect(DATABASE_PATH); cursor = conn.cursor()
        cursor.execute("DELETE FROM conversions WHERE id=?",(item_id,))
        conn.commit(); conn.close()
    except Exception: pass
    return jsonify({"success":True})

@app.route("/api/settings", methods=["POST"])
def update_settings():
    init_session()
    session["settings"].update(request.json); session.modified = True
    return jsonify({"success":True,"settings":session["settings"]})

# ── Boot ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Multilingual AI — Local Model Edition")
    print("=" * 55)
    print(f"  Backend       : {'LOCAL MODELS' if USE_LOCAL_MODELS else 'Gemini API'}")
    print(f"  FFmpeg        : {'✓' if check_ffmpeg() else '✗ (install for audio)'}")
    if USE_LOCAL_MODELS:
        import torch
        print(f"  CUDA          : {'✓ ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else '✗ CPU only'}")
    print("=" * 55)
    app.run(debug=False, host="0.0.0.0", port=8000, threaded=True)