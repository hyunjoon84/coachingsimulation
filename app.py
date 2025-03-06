import os
import asyncio
import aiohttp
import platform
import uuid
import base64
import subprocess
import requests
import json, re
import time
from gtts import gTTS
from flask import Flask, request, jsonify, render_template, session as flask_session
from dotenv import load_dotenv
import speech_recognition as sr
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timezone
from io import BytesIO
import smartsheet

UTC = timezone.utc

if platform.system() == 'Windows':
    FFMPEG_PATH = 'C:/ffmpeg/bin/ffmpeg.exe'
else:
    FFMPEG_PATH = os.getenv('FFMPEG_PATH', '/usr/bin/ffmpeg')

# Load .env
load_dotenv()

# OpenAI API configurations
OPENAI_API_ENDPOINT = "https://api.openai.com/v1/chat/completions"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_MODEL = "gpt-4o-mini"

# Smartsheet API configurations
SMARTSHEET_API_KEY = os.getenv("SMARTSHEET_API_KEY")
SMARTSHEET_SHEET_ID = os.getenv("SMARTSHEET_SHEET_ID")

# FFmpeg path configuration
FFMPEG_PATH = os.getenv("FFMPEG_PATH", "ffmpeg")
if FFMPEG_PATH:
    os.environ["PATH"] = FFMPEG_PATH + ";" + os.environ["PATH"]

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", os.urandom(24).hex())
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("DATABASE_URI", 'sqlite:///coaching_sessions.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Define coaching steps
COACHING_STEPS = [
    "center_together", 
    "clarify_focus", 
    "identify_goal", 
    "develop_action_plan", 
    "gain_commitment", 
    "assess_progress"
]

STEP_DESCRIPTIONS = {
    "center_together": "Centering together involves helping the coachee clear their mind and focus on the present moment.",
    "clarify_focus": "Clarifying the focus involves understanding the specific challenge or issue the coachee wants to address.",
    "identify_goal": "Identifying the goal involves establishing what the coachee wants to achieve from this coaching session.",
    "develop_action_plan": "Developing an action plan involves determining specific steps to achieve the identified goal.",
    "gain_commitment": "Gaining commitment involves ensuring the coachee is motivated and committed to the action plan.",
    "assess_progress": "Assessing progress involves discussing how the coachee will track their progress and how you'll follow up."
}

# Load the coaching content reference material
with open("content_summary.txt", "r", encoding="utf-8") as f:
    CONTENT_SUMMARY = f.read()

# Load step-specific content (optional, kept for consistency)
STEP_CONTENT = {}
for step in COACHING_STEPS:
    try:
        with open(f"content/{step}.txt", "r", encoding="utf-8") as f:
            STEP_CONTENT[step] = f.read()
    except FileNotFoundError:
        STEP_CONTENT[step] = ""

# Database models
class CoachingSession(db.Model):
    id = db.Column(db.String(36), primary_key=True)
    scenario_type = db.Column(db.String(50))
    difficulty = db.Column(db.String(20))
    start_time = db.Column(db.DateTime, default=datetime.now(UTC))
    end_time = db.Column(db.DateTime, nullable=True)
    final_score = db.Column(db.Float, nullable=True)
    final_grade = db.Column(db.String(2), nullable=True)
    strengths = db.Column(db.Text, nullable=True)
    areas_of_improvement = db.Column(db.Text, nullable=True)
    conversation = db.relationship('ConversationMessage', backref='session', lazy=True)

class ConversationMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(36), db.ForeignKey('coaching_session.id'), nullable=False)
    role = db.Column(db.String(20), nullable=False)  # 'coach' or 'employee'
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.now(UTC))
    step = db.Column(db.String(50), nullable=True)  # Current coaching step
    feedback = db.Column(db.Text, nullable=True)  # Feedback for this message

# Create the database and tables
with app.app_context():
    db.create_all()

#######################################################
# Audio Processing Functions
#######################################################
def convert_to_wav(input_file, output_file):
    """Convert audio file to WAV format using ffmpeg."""
    try:
        ffmpeg_cmd = [FFMPEG_PATH, "-y", "-i", input_file, "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", output_file]
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error converting audio: {e}")
        print(f"STDERR: {e.stderr.decode() if e.stderr else 'None'}")
        return False

def generate_speech(text):
    """Convert text to speech using gTTS and return as a data URI without saving to disk."""
    tts = gTTS(text=text, lang='en', slow=False)
    fp = BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    base64_audio = base64.b64encode(fp.read()).decode("utf-8")
    data_uri = f"data:audio/mp3;base64,{base64_audio}"
    return data_uri

def transcribe_audio(base64_audio):
    tmp_dir = "/tmp"
    os.makedirs(tmp_dir, exist_ok=True)  # /tmp 디렉토리 존재 확인
    webm_filename = os.path.join(tmp_dir, f"temp_{uuid.uuid4()}.webm")
    wav_filename = os.path.join(tmp_dir, f"temp_{uuid.uuid4()}.wav")
    try:
        if ',' not in base64_audio:
            return "Audio conversion failed. Please check your audio input and try again."
        audio_bytes = base64.b64decode(base64_audio.split(',')[1])
        with open(webm_filename, 'wb') as f:
            f.write(audio_bytes)
        # 파일이 제대로 생성되었는지 체크
        if not os.path.exists(webm_filename) or os.path.getsize(webm_filename) == 0:
            return "Audio conversion failed. The audio file was not saved correctly."
        print(f"WebM file saved: {webm_filename}, size: {os.path.getsize(webm_filename)} bytes")
        if not convert_to_wav(webm_filename, wav_filename):
            return "Audio conversion failed. Please check your audio input and try again."
        print(f"WAV file saved: {wav_filename}, size: {os.path.getsize(wav_filename)} bytes")
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_filename) as source:
            audio = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio)
                print("Google recognition succeeded:", text)
                return text
            except sr.UnknownValueError:
                print("Google could not understand audio")
                try:
                    text = recognizer.recognize_sphinx(audio)
                    print("Sphinx recognition succeeded:", text)
                    return text
                except sr.UnknownValueError:
                    print("Sphinx could not understand audio")
                    return "Could not understand audio. Please speak clearly and try again."
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return f"Error processing audio: {str(e)}. Please try again."
    finally:
        for file in (webm_filename, wav_filename):
            if os.path.exists(file):
                try:
                    os.remove(file)
                    print(f"Deleted {file}")
                except Exception as e:
                    print(f"Failed to delete {file}: {e}")


#######################################################
# OpenAI API Functions
#######################################################
def call_openai_api(messages, model=DEFAULT_MODEL, temperature=0.7, max_tokens=800):
    """Call OpenAI API with a chat completion request"""
    try:
        if not OPENAI_API_KEY:
            return "Error: OpenAI API key not found in environment variables."
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        max_retries = 3
        retry_delay = 2
        for attempt in range(max_retries):
            try:
                resp = requests.post(OPENAI_API_ENDPOINT, headers=headers, json=payload, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                if "choices" in data and len(data["choices"]) > 0:
                    return data["choices"][0]["message"]["content"]
                else:
                    return "No content returned from API."
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise e
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return f"Error: Could not generate text. {str(e)}"

#######################################################
# Coaching Scenario Generation
#######################################################
def generate_scenario_intro(difficulty, scenario_type):
    """
    Generate a short introduction (3-4 sentences) for a child welfare or child protective services scenario.
    The introduction is from the perspective of a worker who needs coaching.
    """
    system_message = {
        "role": "system", 
        "content": "You are a coaching scenario generator, creating realistic workplace situations in child welfare or child protective services that require coaching."
    }
    user_message = {
        "role": "user", 
        "content": f"""
Create a short introduction (3-4 sentences) from the perspective of a child welfare worker (e.g., a Child Protective Specialist)
who needs coaching. The scenario should focus on child welfare or child protective contexts.

- Difficulty level: {difficulty.capitalize()}
- Scenario type: {scenario_type.capitalize()}
Possible topics include safety assessments, mandated reporting, family engagement, dealing with resistant parents, or provider agency collaboration.
Make it conversational and realistic. The worker should briefly explain their situation and challenge.
"""
    }
    return call_openai_api([system_message, user_message], temperature=0.8)

#######################################################
# Helper Function for Parsing LLM Response
#######################################################
def parse_llm_response(response: str):
    """
    Parse the LLM response to extract the reply and feedback as a dictionary.
    If the response is not in JSON format, return the reply as-is and an empty dict for feedback.
    """
    cleaned_response = response.strip()
    if not cleaned_response.startswith("{"):
        return cleaned_response, {}
    try:
        parsed = json.loads(cleaned_response)
        reply = parsed.get("reply", "I'm not sure how to respond to that.")
        feedback_dict = parsed.get("feedback", {})
        if not isinstance(feedback_dict, dict):
            feedback_dict = {}
        return reply, feedback_dict
    except json.JSONDecodeError:
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response.replace("```json", "", 1)
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response.rsplit("```", 1)[0]
        cleaned_response = cleaned_response.strip()
        try:
            parsed = json.loads(cleaned_response)
            reply = parsed.get("reply", "I'm not sure how to respond to that.")
            feedback_dict = parsed.get("feedback", {})
            if not isinstance(feedback_dict, dict):
                feedback_dict = {}
            return reply, feedback_dict
        except json.JSONDecodeError:
            json_match = re.search(r'\{.*\}', cleaned_response, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group(0))
                    reply = parsed.get("reply", "I'm not sure how to respond to that.")
                    feedback_dict = parsed.get("feedback", {})
                    if not isinstance(feedback_dict, dict):
                        feedback_dict = {}
                    return reply, feedback_dict
                except json.JSONDecodeError:
                    pass
            reply_match = re.search(r'"reply"\s*:\s*"([^"]+)"', cleaned_response)
            feedback_match = re.search(r'"feedback"\s*:\s*(\{.*?\})', cleaned_response, re.DOTALL)
            if reply_match:
                reply = reply_match.group(1)
            else:
                reply = cleaned_response.split('\n')[0][:100] + "..."
            feedback_dict = {}
            if feedback_match:
                try:
                    feedback_dict = json.loads(feedback_match.group(1))
                except:
                    feedback_dict = {}
            return reply, feedback_dict

#######################################################
# Generate Employee Response
#######################################################
def generate_employee_response(session_id, coach_text, current_step):
    """
    Generate only the employee's in-character response based on the conversation history.
    """
    messages = ConversationMessage.query.filter_by(session_id=session_id).order_by(ConversationMessage.timestamp).all()
    conversation_history = []
    for msg in messages:
        if msg.role == "coach":
            conversation_history.append({"role": "user", "content": msg.content})
        else:
            conversation_history.append({"role": "assistant", "content": msg.content})
    
    system_message = {
        "role": "system",
        "content": f"""
You are an employee participating in a coaching session. Respond naturally and in-character to the coach's latest message, considering the current step: {current_step}.
You don't need to be overly formal or coach-like; instead, keep your response straightforward and constructive. If the coach's message seems to stray, briefly acknowledge it but steer the conversation back to practical solutions. 
Return ONLY a string representing your reply, no JSON or additional structure.
"""
    }

    latest_message = {"role": "user", "content": coach_text}
    all_messages = [system_message] + conversation_history + [latest_message]
    
    response = call_openai_api(all_messages, temperature=0.5)
    if isinstance(response, str):
        return response.strip()
    else:
        return "I'm not sure how to respond to that."

#######################################################
# Generate Supervisor Response with GPT-Based Step Detection
#######################################################
def generate_supervisor_response(session_id, coach_text, current_step):
    """
    Generate feedback from the coach's supervisor, determining the current coaching step and evaluating performance.
    Uses GPT to analyze conversation history and provide feedback based on content_summary.txt guidelines.
    """
    messages = ConversationMessage.query.filter_by(session_id=session_id).order_by(ConversationMessage.timestamp).all()
    conversation_history = []
    for msg in messages:
        conversation_history.append({"role": "user" if msg.role == "coach" else "assistant", "content": msg.content})
    
    system_message = {
        "role": "system",
        "content": f"""
You are the coach's supervisor with full knowledge of the coaching guidelines from the content_summary document. Your task is to:
1. Analyze the entire conversation history and the coach's latest message to determine the current coaching step from these options: {', '.join(COACHING_STEPS)}.
2. Evaluate the coach's performance in the current coaching process. Provide detailed, concrete feedback strictly based on the benchmarks and criteria outlined in the content_summary.txt document. If you observe that the coach's transition to the next step is premature or incomplete, note this explicitly.
3. Evaluate the coach's skills by referencing specific guidelines from the content_summary.txt document. Do not offer generic compliments; instead, provide observations that are directly tied to the active listening techniques, questioning strategies, and other benchmarks detailed in the document.
4. Where applicable, reference specific coaching questions or guidelines from the content_summary.txt to ensure that your feedback is deeply contextual and aligned with the established standards.


The coaching steps are:
- center_together: Help the coachee clear their mind and focus.
- clarify_focus: Understand the coachee’s specific challenge or issue.
- identify_goal: Establish what the coachee wants to achieve.
- develop_action_plan: Determine specific steps to achieve the goal.
- gain_commitment: Ensure the coachee is motivated to follow the plan.
- assess_progress: Discuss tracking progress and follow-up.

*Coaching Questions (for context):
1. Center Together:
  • How would you like to center?
  • What does centering look like to you?
  • How ready are you feeling after centering?
2. Clarify the Focus:
  • What is the key problem or challenge for you?
  • What is significant about this for you right now?
  • How does this relate to other issues you are facing?
3. Identify the Goal:
  • What outcomes would be ideal here?
  • What impact do you want to have?
  • How will you know you have been successful?
4. Develop an Action Plan:
  • What is a specific action you can take toward this?
  • What might be the first step?
  • What support do you need?
5. Gain Commitment:
  • How committed are you to taking these steps on a scale of 1 to 10?
  • What might prevent you from taking these steps?
6. Assess Progress:
  • How will you assess your progress toward your goal(s)?
  • How can I better support you in achieving your goal?
  • How helpful was this conversation for you?

Return ONLY valid JSON with one top-level key: "feedback", containing:
- "current_step": The detected step (string).
- "Coaching_Process": {{
    "comments": [an array of strings containing your observations and suggestions about the coaching process, with each comment directly tied to specific guidelines from content_summary.txt],
    "step_complete": "complete" | "premature" | "partial" | "incomplete"
      // You can expand or rename these values as needed
  }}
- "Coaching_Skills": {{
    "comments": [array of strings with feedback on the coach’s skills based strictly on content_summary guidelines]
  }}

Example:
{{
  "feedback": {{
    "current_step": "clarify_focus",
    "Coaching_Process": {{
      "comments": ["The coach asked about the issue but didn’t fully clarify it.", "More probing needed before moving to goals."],
      "step_complete":  "premature"
       /* Possible values:
         - "complete": The step was fully addressed
         - "premature": Moving on too soon (the step still needs more exploration)
         - "partial": Some progress was made, but it's not fully complete
         - "incomplete": The step was barely addressed or not addressed at all
      */     
    }},
    "Coaching_Skills": {{
      "comments": ["The coach did not sufficiently reference the active listening techniques described in the content_summary.", "The response lacks specific alignment with the established coaching benchmarks."]
    }}
  }}
}}

Conversation history:
{json.dumps(conversation_history, indent=2)}

Latest coach message: '{coach_text}'
"""
    }

    all_messages = [system_message]
    
    # First API call with strict temperature
    response = call_openai_api(all_messages, temperature=0.1)
    print("Raw supervisor response (1st):", response)
    _, feedback_dict = parse_llm_response(response)
    
    def is_valid_feedback(fdict):
        return (
            isinstance(fdict, dict)
            and "current_step" in fdict
            and "Coaching_Process" in fdict
            and "Coaching_Skills" in fdict
            and isinstance(fdict["Coaching_Process"], dict)
            and isinstance(fdict["Coaching_Skills"], dict)
            and "comments" in fdict["Coaching_Process"]
            and "step_complete" in fdict["Coaching_Process"]
            and "comments" in fdict["Coaching_Skills"]
            and len(fdict["Coaching_Process"]["comments"]) > 0
            and len(fdict["Coaching_Skills"]["comments"]) > 0
        )
    
    if is_valid_feedback(feedback_dict):
        feedback_list = [
            f"Current Step: {feedback_dict['current_step']}",
            f"Coaching Process (Complete: {feedback_dict['Coaching_Process']['step_complete']}): {', '.join(feedback_dict['Coaching_Process']['comments'])}",
            f"Coaching Skills: {', '.join(feedback_dict['Coaching_Skills']['comments'])}"
        ]
        return feedback_list, feedback_dict["current_step"]
    
    # Fallback attempt 2
    second_system = {
        "role": "system",
        "content": "You are a JSON converter. Transform the following text into the required JSON structure with 'feedback'."
    }
    second_prompt = f"""
Transform the following text into valid JSON with:
- "feedback": {{
    "current_step": "{current_step}",
    "Coaching_Process": {{
      "comments": [...],
      "step_complete": true/false
    }},
    "Coaching_Skills": {{
      "comments": [...]
    }}
}}

Text:
{response}
"""
    second_messages = [second_system, {"role": "user", "content": second_prompt}]
    second_response = call_openai_api(second_messages, temperature=0.0)
    print("Raw GPT-4o-mini fallback response:", second_response)
    _, second_feedback_dict = parse_llm_response(second_response)
    
    if is_valid_feedback(second_feedback_dict):
        feedback_list = [
            f"Current Step: {second_feedback_dict['current_step']}",
            f"Coaching Process (Complete: {second_feedback_dict['Coaching_Process']['step_complete']}): {', '.join(second_feedback_dict['Coaching_Process']['comments'])}",
            f"Coaching Skills: {', '.join(second_feedback_dict['Coaching_Skills']['comments'])}"
        ]
        return feedback_list, second_feedback_dict["current_step"]
    
    # Fallback attempt 3
    third_system = {
        "role": "system",
        "content": "You are a JSON converter. Transform the following text into the required JSON structure with 'feedback'."
    }
    third_prompt = f"""
Transform the following text into valid JSON with:
- "feedback": {{
    "current_step": "{current_step}",
    "Coaching_Process": {{
      "comments": [...],
      "step_complete": true/false
    }},
    "Coaching_Skills": {{
      "comments": [...]
    }}
}}

Text:
{response}
"""
    third_messages = [third_system, {"role": "user", "content": third_prompt}]
    third_response = call_openai_api(third_messages, model="gpt-3.5-turbo", temperature=0.0)
    print("Raw GPT-3.5-turbo fallback response:", third_response)
    _, third_feedback_dict = parse_llm_response(third_response)
    
    if is_valid_feedback(third_feedback_dict):
        feedback_list = [
            f"Current Step: {third_feedback_dict['current_step']}",
            f"Coaching Process (Complete: {third_feedback_dict['Coaching_Process']['step_complete']}): {', '.join(third_feedback_dict['Coaching_Process']['comments'])}",
            f"Coaching Skills: {', '.join(third_feedback_dict['Coaching_Skills']['comments'])}"
        ]
        return feedback_list, third_feedback_dict["current_step"]
    
    # Final fallback
    return [
        f"Current Step: {current_step}",
        "Coaching Process (Complete: False): Unable to evaluate properly due to response issues.",
        "Coaching Skills: Please review the coach's approach for further assessment."
    ], current_step

#######################################################
# Generate Final (or Partial) Evaluation
#######################################################
def generate_final_evaluation(session_id):
    """Generate a comprehensive final evaluation of the coaching session."""
    messages = ConversationMessage.query.filter_by(session_id=session_id).order_by(ConversationMessage.timestamp).all()
    coach_messages = [msg for msg in messages if msg.role == "coach"]
    if len(coach_messages) < 3:
        return {
            "percentage": 0,
            "grade": "F",
            "strengths": ["No significant coaching interaction occurred."],
            "areas_of_improvement": ["Coach needs to engage more actively in the session."]
        }
    
    # 어떤 단계가 다뤄졌는지 확인
    covered_steps = set()
    for msg in messages:
        if msg.step and msg.step in COACHING_STEPS:
            covered_steps.add(msg.step)
    
    missing_steps = [step for step in COACHING_STEPS if step not in covered_steps]
    
    conversation_text = ""
    for msg in messages:
        if msg.role == "coach":
            conversation_text += f"Coach: {msg.content}\n"
        else:
            conversation_text += f"Employee: {msg.content}\n"
            
    system_message = {
        "role": "system",
        "content": f"""
You are an expert coach evaluator. Focus on these instructions:
1. Check if all required coaching steps were addressed. The required steps are: {', '.join(COACHING_STEPS)}
2. Based on the conversation analysis, the following steps were MISSING: {', '.join(missing_steps) if missing_steps else 'None'}
3. You MUST deduct 10 points for each missing coaching step. For {len(missing_steps)} missing steps, deduct {len(missing_steps) * 10} points.
4. You MUST mention each missing step in the "areas_of_improvement" array.
5. Evaluate coach's skills: listening, questioning, empathy, clarity, relevance based on content_summary.txt.
6. Give a numeric score 0-100, accounting for the mandatory deductions for missing steps. Then assign A-F grade.
7. Provide an array "strengths" (3-5 items) and "areas_of_improvement" (at least {len(missing_steps)} items for missing steps, plus 1-2 other areas if needed).

You MUST Return JSON with keys: "percentage", "grade", "strengths", "areas_of_improvement" (all arrays except grade is string, percentage is number).

Format your response as JSON:
{{
  "percentage": 85,
  "grade": "B",
  "strengths": ["Strength 1", "Strength 2", "Strength 3"],
  "areas_of_improvement": ["Did not complete step X", "Did not address step Y", "Area 3"]
}}
"""
    }
    
    user_message = {
        "role": "user",
        "content": f"""
Here is the complete coaching conversation to evaluate:

{conversation_text}

Please provide a final evaluation as specified.
Remember, the following coaching steps were MISSING: {', '.join(missing_steps) if missing_steps else 'None'}
You MUST deduct 10 points for each missing step ({len(missing_steps) * 10} points total) and include each missing step in the areas_of_improvement.
"""
    }
    
    response = call_openai_api([system_message, user_message], temperature=0.3, max_tokens=1000)

    def normalize_text(txt: str) -> str:
        # Lowercase, replace underscores with spaces, remove punctuation
        txt = txt.lower().replace("_", " ")
        # optionally remove punctuation (commas, periods, etc.)
        txt = re.sub(r"[^\w\s]", "", txt)
        return txt
        
    try:
        parsed = json.loads(response)
        
        # 누락된 단계가 있는데 개선 영역에 누락된 단계가 포함되어 있지 않으면 강제로 추가
        if missing_steps:
             # Ensure the 'areas_of_improvement' field exists
            if "areas_of_improvement" not in parsed or not parsed["areas_of_improvement"]:
                parsed["areas_of_improvement"] = []
             # -- 1) Build a set of step names that GPT might have already mentioned in some form --
            gpt_improvement_set = set()
            for item in parsed["areas_of_improvement"]:
            # Look for any missing step name in the text of the item
                normalized_item = normalize_text(item)
                for step_name in missing_steps:
                    normalized_step = normalize_text(step_name)
                    # For example, if item = "Did not address step: develop_action_plan" 
                    # and step_name = "develop_action_plan", we'll detect it
                    if normalized_step in normalized_item:
                        gpt_improvement_set.add(step_name)

            # -- 2) Only add the forced line if GPT did NOT already mention that step --
            for step in missing_steps:
                if step not in gpt_improvement_set:
                    improvement_text = f"Did not complete the '{step}' coaching step"
                    # This final check ensures we don't double-append if the exact text already exists
                    if improvement_text not in parsed["areas_of_improvement"]:
                        parsed["areas_of_improvement"].append(improvement_text)
            
            # 점수 강제 조정 (누락된 단계당 10점 차감)
            if "percentage" in parsed:
                max_score = 100
                deduction = len(missing_steps) * 10
                parsed["percentage"] = max(0, min(max_score - deduction, parsed["percentage"]))
                
                # 등급 업데이트
                if parsed["percentage"] >= 90:
                    parsed["grade"] = "A"
                elif parsed["percentage"] >= 80:
                    parsed["grade"] = "B"
                elif parsed["percentage"] >= 70:
                    parsed["grade"] = "C"
                elif parsed["percentage"] >= 60:
                    parsed["grade"] = "D"
                else:
                    parsed["grade"] = "F"
        
    except:
        try:
            json_match = re.search(r'\{.*"percentage".*"grade".*"strengths".*"areas_of_improvement".*\}', response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                
                # 여기서도 누락된 단계 확인 및 강제 추가
                if missing_steps:
                    if "areas_of_improvement" not in parsed or not parsed["areas_of_improvement"]:
                        parsed["areas_of_improvement"] = []
                    
                    existing_improvements = set(parsed["areas_of_improvement"])
                    for step in missing_steps:
                        improvement_text = f"Did not complete the '{step}' coaching step"
                        if improvement_text not in existing_improvements:
                            parsed["areas_of_improvement"].append(improvement_text)
                    
                    # 점수 강제 조정
                    if "percentage" in parsed:
                        max_score = 100
                        deduction = len(missing_steps) * 10
                        parsed["percentage"] = max(0, min(max_score - deduction, parsed["percentage"]))
                        
                        # 등급 업데이트
                        if parsed["percentage"] >= 90:
                            parsed["grade"] = "A"
                        elif parsed["percentage"] >= 80:
                            parsed["grade"] = "B"
                        elif parsed["percentage"] >= 70:
                            parsed["grade"] = "C"
                        elif parsed["percentage"] >= 60:
                            parsed["grade"] = "D"
                        else:
                            parsed["grade"] = "F"
            else:
                # 기본 응답 생성 (누락된 단계 반영)
                deduction = len(missing_steps) * 10
                percentage = max(0, 70 - deduction)  # 기본 70점에서 차감
                
                # 등급 계산
                if percentage >= 90:
                    grade = "A"
                elif percentage >= 80:
                    grade = "B"
                elif percentage >= 70:
                    grade = "C"
                elif percentage >= 60:
                    grade = "D"
                else:
                    grade = "F"
                
                areas_of_improvement = []
                for step in missing_steps:
                    areas_of_improvement.append(f"Did not complete the '{step}' coaching step")
                
                if not areas_of_improvement:
                    areas_of_improvement = ["Structure coaching process better", "Ask more open-ended questions"]
                
                parsed = {
                    "percentage": percentage,
                    "grade": grade,
                    "strengths": ["Communication skills", "Building rapport"],
                    "areas_of_improvement": areas_of_improvement
                }
        except:
            # 가장 기본적인 응답 (누락된 단계 반영)
            deduction = len(missing_steps) * 10
            percentage = max(0, 70 - deduction)  # 기본 70점에서 차감
            
            # 등급 계산
            if percentage >= 90:
                grade = "A"
            elif percentage >= 80:
                grade = "B"
            elif percentage >= 70:
                grade = "C"
            elif percentage >= 60:
                grade = "D"
            else:
                grade = "F"
            
            areas_of_improvement = []
            for step in missing_steps:
                areas_of_improvement.append(f"Did not complete the '{step}' coaching step")
            
            if not areas_of_improvement:
                areas_of_improvement = ["Structure coaching process better", "Ask more open-ended questions"]
            
            parsed = {
                "percentage": percentage,
                "grade": grade,
                "strengths": ["Communication skills", "Building rapport"],
                "areas_of_improvement": areas_of_improvement
            }
    
    return parsed

#######################################################
# Smartsheet Integration Functions
#######################################################
def save_session_to_smartsheet(session_id):
    session_obj = db.session.get(CoachingSession, session_id)
    if not session_obj:
        print("Session not found.")
        return
    messages = ConversationMessage.query.filter_by(session_id=session_id).order_by(ConversationMessage.timestamp).all()
    conversation_text = ""
    for msg in messages:
        conversation_text += f"{msg.role.capitalize()}: {msg.content}\n"
    try:
        strengths = json.loads(session_obj.strengths) if session_obj.strengths else []
    except:
        strengths = []
    try:
        improvements = json.loads(session_obj.areas_of_improvement) if session_obj.areas_of_improvement else []
    except:
        improvements = []
    feedback_summary = "Strengths: " + ", ".join(strengths) + " | Areas for Improvement: " + ", ".join(improvements)
    coaching_feedback = ""
    for msg in messages:
        feedback_list = []
        if msg.role == "employee" and msg.feedback:
            try:
                feedback_list = json.loads(msg.feedback)
            except json.JSONDecodeError:
                feedback_list = []
        for f in feedback_list:
            coaching_feedback += f + "\n"
    smartsheet_client = smartsheet.Smartsheet(SMARTSHEET_API_KEY)
    smartsheet_client.errors_as_exceptions(True)
    new_row = smartsheet.models.Row()
    new_row.to_top = True
    cell0 = smartsheet.models.Cell()
    cell0.column_id = get_column_id("ID")
    cell0.value = session_obj.id
    cell1 = smartsheet.models.Cell()
    cell1.column_id = get_column_id("Difficulty Level")
    cell1.value = session_obj.difficulty
    cell2 = smartsheet.models.Cell()
    cell2.column_id = get_column_id("Scenario Type")
    cell2.value = session_obj.scenario_type
    cell3 = smartsheet.models.Cell()
    cell3.column_id = get_column_id("Conversation")
    cell3.value = conversation_text
    cell4 = smartsheet.models.Cell()
    cell4.column_id = get_column_id("Coaching Feedback")
    cell4.value = coaching_feedback
    cell5 = smartsheet.models.Cell()
    cell5.column_id = get_column_id("Feedback Summary")
    cell5.value = feedback_summary
    cell6 = smartsheet.models.Cell()
    cell6.column_id = get_column_id("Score")
    cell6.value = session_obj.final_score
    cell7 = smartsheet.models.Cell()
    cell7.column_id = get_column_id("Grade")
    cell7.value = session_obj.final_grade
    new_row.cells = [cell0, cell1, cell2, cell3, cell4, cell5, cell6, cell7]
    try:
        response = smartsheet_client.Sheets.add_rows(SMARTSHEET_SHEET_ID, [new_row])
        print("Session saved to Smartsheet.")
    except Exception as e:
        print("Error saving to Smartsheet:", e)

def get_column_id(column_name):
    column_mapping = {
        "ID": 6891979856367492,
        "Difficulty Level": 4480157681405828,
        "Scenario Type": 8983757308776324,
        "Conversation": 4532934239539076,
        "Coaching Feedback": 29334612168580,
        "Feedback Summary": 7250363805814660,
        "Score": 2746764178444164,
        "Grade": 4998563992129412
    }
    return column_mapping.get(column_name)

#######################################################
# Flask Routes
#######################################################
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/start-simulation", methods=["POST"])
def start_simulation():
    data = request.json
    difficulty = data.get("difficulty", "beginner")
    scenario_type = data.get("scenario_type", "performance")
    intro_text = generate_scenario_intro(difficulty, scenario_type)
    session_id = str(uuid.uuid4())
    new_session = CoachingSession(
        id=session_id,
        scenario_type=scenario_type,
        difficulty=difficulty
    )
    db.session.add(new_session)
    intro_message = ConversationMessage(
        session_id=session_id,
        role="employee",
        content=intro_text,
        step="center_together"
    )
    db.session.add(intro_message)
    db.session.commit()
    audio_file = generate_speech(intro_text)
    return jsonify({
        "session_id": session_id,
        "text": intro_text,
        "audio": audio_file,
        "scenario_name": f"{scenario_type.capitalize()} ({difficulty.capitalize()})",
        "current_step": "center_together",
        "step_description": STEP_DESCRIPTIONS["center_together"]
    })

@app.route("/api/respond", methods=["POST"])
def respond():
    data = request.json
    session_id = data.get("session_id")
    audio_data = data.get("audio")
    text = data.get("text", "")
    session = db.session.get(CoachingSession, session_id)
    if not session:
        return jsonify({"error": "Invalid session"}), 400
    if session.end_time:
        return jsonify({"error": "Session is already complete"}), 400
    if audio_data:
        user_text = transcribe_audio(audio_data)
    else:
        user_text = text.strip()
    # 수정: 음성 인식 실패 혹은 에러 발생 시 텍스트 입력 허용 플래그 추가
    if not user_text or "Could not understand audio" in user_text or "Error processing audio" in user_text:
        return jsonify({
            "error": user_text,
            "text": "I couldn't hear you clearly. Please try again or type your response.",
            "audio": generate_speech("I couldn't hear you clearly. Please try again or type your response."),
            "allow_text_input": True
        }), 200
    last_message = ConversationMessage.query.filter_by(session_id=session_id).order_by(ConversationMessage.timestamp.desc()).first()
    current_step = last_message.step if last_message else "center_together"
    coach_message = ConversationMessage(
        session_id=session_id,
        role="coach",
        content=user_text,
        step=current_step  # Initial step, will be updated by supervisor
    )
    db.session.add(coach_message)
    db.session.commit()
    
    # Generate employee's reply
    reply_text = generate_employee_response(session_id, user_text, current_step)
    
    # Generate supervisor's feedback and detect step
    feedback_list, detected_step = generate_supervisor_response(session_id, user_text, current_step)
    
    employee_message = ConversationMessage(
        session_id=session_id,
        role="employee",
        content=reply_text,
        step=detected_step,  # Use GPT-detected step
        feedback=json.dumps(feedback_list)
    )
    db.session.add(employee_message)
    db.session.commit()
    audio_file = generate_speech(reply_text)
    completed_steps = {msg.step for msg in ConversationMessage.query.filter_by(session_id=session_id).all()}
    is_complete = False
    final_score = None
    if set(COACHING_STEPS).issubset(completed_steps) and detected_step == COACHING_STEPS[-1]:
        is_complete = True
        evaluation = generate_final_evaluation(session_id)
        session.end_time = datetime.now(UTC)
        session.final_score = evaluation["percentage"]
        session.final_grade = evaluation["grade"]
        session.strengths = json.dumps(evaluation["strengths"])
        session.areas_of_improvement = json.dumps(evaluation["areas_of_improvement"])
        db.session.commit()
        final_score = evaluation
    partial_score = None
    if not is_complete:
        partial_score = generate_final_evaluation(session_id)
    return jsonify({
        "text": reply_text,
        "audio": audio_file,
        "evaluation": {"feedback": feedback_list},
        "is_complete": is_complete,
        "final_score": final_score,
        "partial_score": partial_score,
        "current_step": detected_step,
        "step_description": STEP_DESCRIPTIONS[detected_step],
        "step_advanced": detected_step != current_step,
        "coach_input": user_text
    })

@app.route("/api/skip-to-end", methods=["POST"])
def skip_to_end():
    data = request.json
    session_id = data.get("session_id")
    session = db.session.get(CoachingSession, session_id)
    if not session:
        return jsonify({"error": "Invalid session"}), 400
    if session.end_time:
        return jsonify({"error": "Session is already complete"}), 400
    evaluation = generate_final_evaluation(session_id)
    session.end_time = datetime.now(UTC)
    session.final_score = evaluation["percentage"]
    session.final_grade = evaluation["grade"]
    session.strengths = json.dumps(evaluation["strengths"])
    session.areas_of_improvement = json.dumps(evaluation["areas_of_improvement"])
    db.session.commit()
    save_session_to_smartsheet(session_id)
    return jsonify({
        "message": "Session completed successfully.",
        "final_score": evaluation
    })

@app.route("/api/sessions", methods=["GET"])
def get_sessions():
    sessions = CoachingSession.query.filter(CoachingSession.end_time.isnot(None)).order_by(CoachingSession.end_time.desc()).all()
    result = []
    for session in sessions:
        result.append({
            "id": session.id,
            "scenario_type": session.scenario_type,
            "difficulty": session.difficulty,
            "start_time": session.start_time.isoformat(),
            "end_time": session.end_time.isoformat() if session.end_time else None,
            "final_score": session.final_score,
            "final_grade": session.final_grade,
            "strengths": json.loads(session.strengths) if session.strengths else [],
            "areas_of_improvement": json.loads(session.areas_of_improvement) if session.areas_of_improvement else []
        })
    return jsonify(result)

@app.route("/api/session/<session_id>", methods=["GET"])
def get_session_details(session_id):
    session = db.session.get(CoachingSession, session_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404
    messages = ConversationMessage.query.filter_by(session_id=session_id).order_by(ConversationMessage.timestamp).all()
    message_list = []
    for msg in messages:
        message_list.append({
            "role": msg.role,
            "content": msg.content,
            "timestamp": msg.timestamp.isoformat(),
            "step": msg.step,
            "feedback": json.loads(msg.feedback) if msg.feedback else None
        })
    result = {
        "id": session.id,
        "scenario_type": session.scenario_type,
        "difficulty": session.difficulty,
        "start_time": session.start_time.isoformat(),
        "end_time": session.end_time.isoformat() if session.end_time else None,
        "final_score": session.final_score,
        "final_grade": session.final_grade,
        "strengths": json.loads(session.strengths) if session.strengths else [],
        "areas_of_improvement": json.loads(session.areas_of_improvement) if session.areas_of_improvement else [],
        "messages": message_list
    }
    return jsonify(result)

@app.route("/api/ffmpeg-test", methods=["GET"])
def ffmpeg_test():
    try:
        result = subprocess.run([FFMPEG_PATH, "-version"], capture_output=True, text=True, check=True)
        print(f"FFmpeg version: {result.stdout}")
        return jsonify({"ffmpeg_version": result.stdout})
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e.stderr}")
        return jsonify({"error": str(e.stderr)}), 500
    
if __name__ == "__main__":
    os.makedirs("static/audio", exist_ok=True)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", debug=True, port=port)
