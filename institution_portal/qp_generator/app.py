import os
import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

# ------------------ Config ------------------
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
QUESTIONS_CSV = os.path.join(DATA_DIR, "questions.csv")
ALLOWED_EXTENSIONS = {'csv'}

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.secret_key = "qp-generator-secret"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# ------------------ Models ------------------
@dataclass
class Question:
    id: str
    subject: str
    unit: str
    topic: str
    question_text: str
    question_type: str
    difficulty: str
    marks: int

# ------------------ Helpers ------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_questions(filepath=None) -> List[Question]:
    csv_path = filepath or QUESTIONS_CSV
    questions: List[Question] = []
    if not os.path.exists(csv_path):
        return questions
    
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                q = Question(
                    id=(row.get("id") or "").strip(),
                    subject=(row.get("subject") or "").strip(),
                    unit=(row.get("unit") or "").strip(),
                    topic=(row.get("topic") or "").strip(),
                    question_text=(row.get("question_text") or "").strip(),
                    question_type=(row.get("question_type") or "").strip(),
                    difficulty=(row.get("difficulty") or "").strip().title(),
                    marks=int((row.get("marks") or "0").strip() or 0),
                )
            except Exception:
                continue
            if q.subject and q.question_text and q.marks > 0:
                questions.append(q)
    return questions

def split_by_marks(pool: List[Question]) -> Dict[int, List[Question]]:
    """Group questions by marks"""
    marks_pools = defaultdict(list)
    for q in pool:
        marks_pools[q.marks].append(q)
    return dict(marks_pools)

def select_by_marks_count(
    marks_pools: Dict[int, List[Question]],
    marks_config: List[Tuple[int, int]]  # [(marks, count), ...]
) -> List[Question]:
    """
    Select questions based on marks and count specification
    marks_config: [(4, 5), (6, 5), (10, 1)] means 5×4m, 5×6m, 1×10m
    """
    selected = []
    used_ids = set()
    
    for marks, count in marks_config:
        available = [q for q in marks_pools.get(marks, []) if q.id not in used_ids]
        if len(available) < count:
            flash(f"⚠️ Warning: Only {len(available)} questions available for {marks} marks (need {count})")
        
        random.shuffle(available)
        selected_batch = available[:count]
        
        for q in selected_batch:
            selected.append(q)
            used_ids.add(q.id)
    
    return selected

def group_by_unit(questions: List[Question]) -> Dict[str, List[Question]]:
    grouped: Dict[str, List[Question]] = defaultdict(list)
    for q in questions:
        key = q.unit or "No Unit"
        grouped[key].append(q)
    return dict(sorted(grouped.items()))

def build_plain_text(
    subject: str,
    paper_title: str,
    total_marks: int,
    duration: int,
    instructions: str,
    questions: List[Question],  # Changed from grouped to flat list
) -> str:
    lines = []
    lines.append("="*70)
    lines.append(paper_title.center(70))
    lines.append("="*70)
    lines.append(f"Subject: {subject}".center(70))
    lines.append(f"Total Marks: {total_marks} | Duration: {duration} minutes".center(70))
    lines.append("="*70)
    lines.append("")
    
    if instructions:
        lines.append("INSTRUCTIONS:")
        lines.append(instructions)
        lines.append("")
    
    lines.append("-"*70)
    lines.append("")
    
    # Just list questions without unit grouping
    for idx, q in enumerate(questions, 1):
        lines.append(f"Q{idx}. [{q.marks} marks] {q.question_text}")
        lines.append("")
    
    lines.append("="*70)
    lines.append("END OF QUESTION PAPER".center(70))
    lines.append("="*70)
    return "\n".join(lines)


# ------------------ Routes ------------------
@app.route("/", methods=["GET"])
def index():
    questions = load_questions()
    subjects = sorted({q.subject for q in questions})
    
    if not subjects:
        flash("No subjects found. Please upload a question bank CSV.")
    
    stats = {}
    marks_distribution = {}
    for subj in subjects:
        subj_questions = [q for q in questions if q.subject == subj]
        stats[subj] = len(subj_questions)
        marks_distribution[subj] = {}
        for q in subj_questions:
            marks_distribution[subj][q.marks] = marks_distribution[subj].get(q.marks, 0) + 1
    
    return render_template("index.html", subjects=subjects, stats=stats, marks_distribution=marks_distribution)

@app.route("/upload", methods=["POST"])
def upload_file():
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Validate and copy to data folder
        try:
            questions = load_questions(filepath)
            if not questions:
                flash('❌ No valid questions found in the CSV file')
                return redirect(url_for('index'))
            
            # Copy to main data folder
            import shutil
            shutil.copy(filepath, QUESTIONS_CSV)
            flash(f'✅ Successfully uploaded {len(questions)} questions!')
        except Exception as e:
            flash(f'❌ Error processing file: {str(e)}')
        
        return redirect(url_for('index'))
    else:
        flash('❌ Invalid file type. Please upload a CSV file.')
        return redirect(url_for('index'))

@app.route("/generate", methods=["POST"])
def generate_paper():
    subject = (request.form.get("subject") or "").strip()
    paper_title = (request.form.get("paper_title") or "").strip() or "Question Paper"
    duration = int(request.form.get("duration") or "90")
    instructions = (request.form.get("instructions") or "").strip()
    
    # Parse marks configuration
    marks_config = []
    try:
        marks_4_count = int(request.form.get("marks_4_count") or "0")
        marks_6_count = int(request.form.get("marks_6_count") or "0")
        marks_10_count = int(request.form.get("marks_10_count") or "0")
        
        if marks_4_count > 0:
            marks_config.append((4, marks_4_count))
        if marks_6_count > 0:
            marks_config.append((6, marks_6_count))
        if marks_10_count > 0:
            marks_config.append((10, marks_10_count))
        
        if not marks_config:
            flash("Please specify at least one question type")
            return redirect(url_for("index"))
    except ValueError:
        flash("Invalid numeric values for question counts")
        return redirect(url_for("index"))
    
    # Load questions and filter by subject
    questions = load_questions()
    pool = [q for q in questions if q.subject == subject]
    
    if not pool:
        flash(f"No questions found for subject '{subject}'.")
        return redirect(url_for("index"))
    
    # Split by marks and select
    marks_pools = split_by_marks(pool)
    selected_all = select_by_marks_count(marks_pools, marks_config)
    
    if not selected_all:
        flash("Could not select any questions. Please check question bank.")
        return redirect(url_for("index"))
    
    # Calculate totals
    total_marks = sum(q.marks for q in selected_all)
    
    # Build plain text WITHOUT unit grouping
    plain_text = build_plain_text(
        subject=subject,
        paper_title=paper_title,
        total_marks=total_marks,
        duration=duration,
        instructions=instructions,
        questions=selected_all,  # Pass flat list instead of grouped
    )
    
    # Calculate distribution
    distribution = {}
    for marks, count in marks_config:
        actual = len([q for q in selected_all if q.marks == marks])
        distribution[marks] = {"target": count, "actual": actual}
    
    # Group by unit for preview display only
    grouped = group_by_unit(selected_all)
    
    return render_template(
        "preview.html",
        subject=subject,
        paper_title=paper_title,
        total_marks=total_marks,
        duration=duration,
        instructions=instructions,
        grouped=grouped,  # For preview details section
        questions=selected_all,  # For simple list display
        plain_text=plain_text,
        num_questions=len(selected_all),
        distribution=distribution,
    )

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
