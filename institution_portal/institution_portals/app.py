# app.py
import os
from functools import wraps
from typing import Dict, List, Tuple, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from datetime import datetime 
import openpyxl
import pandas as pd
from sqlalchemy import func, case
from flask import (
    Flask, render_template, redirect, url_for, request,
    flash, abort, send_from_directory, jsonify
)
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from flask_login import (
    LoginManager, login_user, logout_user,
    login_required, current_user, UserMixin
)
from flask_talisman import Talisman
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from config import Config
from google import genai

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
STUDENTS_CSV = os.path.join(DATA_DIR, "students.csv")
CLASSES_CONFIG_CSV = os.path.join(DATA_DIR, "classes_config.csv")
HALL_CONFIG_CSV = os.path.join(DATA_DIR, "hall_config.csv")
os.makedirs(DATA_DIR, exist_ok=True)



def read_students_from_csv(path: str) -> Dict[str, List[str]]:
    from collections import defaultdict
    by_class = defaultdict(list)
    if not os.path.exists(path):
        return {}
    import csv
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames and "class" in (h.lower() for h in reader.fieldnames):
            lower_map = {name.lower(): name for name in reader.fieldnames}
            cls_key = lower_map.get("class")
            reg_key = (
                lower_map.get("reg_no")
                or lower_map.get("reg")
                or lower_map.get("regno")
            )
            if not reg_key:
                raise ValueError("students CSV must have a reg_no/reg column")
            for row in reader:
                cls = row.get(cls_key, "").strip()
                reg = row.get(reg_key, "").strip()
                if cls and reg:
                    by_class[cls].append(reg)
        else:
            f.seek(0)
            r = csv.reader(f)
            for row in r:
                if not row:
                    continue
                if len(row) >= 2:
                    cls = str(row[0]).strip()
                    reg = str(row[1]).strip()
                    if cls and reg:
                        by_class[cls].append(reg)
    return dict(by_class)


def read_classes_config(path: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    if not os.path.exists(path):
        return out
    import csv
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames and "class" in (h.lower() for h in reader.fieldnames):
            lower_map = {name.lower(): name for name in reader.fieldnames}
            cls_key = lower_map.get("class")
            ben_key = (
                lower_map.get("allocated_benches")
                or lower_map.get("benches")
                or lower_map.get("allocated")
            )
            if ben_key:
                for row in reader:
                    cls = row.get(cls_key, "").strip()
                    ben_raw = row.get(ben_key, "").strip()
                    try:
                        ben = int(ben_raw)
                    except Exception:
                        continue
                    if cls and ben > 0:
                        out[cls] = ben
                return out
        f.seek(0)
        r = csv.reader(f)
        for row in r:
            if not row or len(row) < 2:
                continue
            try:
                cls = str(row[0]).strip()
                ben = int(row[1])
            except Exception:
                continue
            if cls and ben > 0:
                out[cls] = ben
    return out

def seating_grid_to_csv_bytes_from_rows(
    rows: List[Tuple[str, int, int, str, str]]
) -> bytes:
    import csv, io
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["hall_name", "bench", "seat_no", "class", "reg_no"])
    for hall_name, bench_no, seat_no, cls, reg in rows:
        writer.writerow([hall_name, bench_no, seat_no, cls, reg])
    return out.getvalue().encode("utf-8")



def read_halls(path: str) -> List[Tuple[str, int, int]]:
    halls: List[Tuple[str, int, int]] = []
    if not os.path.exists(path):
        return halls
    import csv
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames:
            lower_map = {name.lower(): name for name in reader.fieldnames}
            hall_key = lower_map.get("hall_name")
            seats_key = lower_map.get("seats_per_bench")
            benches_key = lower_map.get("allocated_benches")
            if hall_key and seats_key and benches_key:
                for row in reader:
                    hall_name = row.get(hall_key, "").strip()
                    if not hall_name:
                        continue
                    seats_raw = row.get(seats_key, "").strip()
                    benches_raw = row.get(benches_key, "").strip()
                    try:
                        seats = int(seats_raw) if seats_raw else 2
                    except Exception:
                        seats = 2
                    try:
                        benches = int(benches_raw) if benches_raw else 0
                    except Exception:
                        benches = 0
                    if benches > 0:
                        halls.append((hall_name, seats, benches))
    return halls



# -------------------------------------------------------------------
# Gemini config
# -------------------------------------------------------------------
GEMINI_API_KEY = " put your key"  # put your key
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
GEMINI_MODEL = "models/gemini-2.5-flash"
# -------------------------------------------------------------------
# LangChain LLM wrapper for Gemini
# -------------------------------------------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
    max_output_tokens=256,
    google_api_key=GEMINI_API_KEY,
)

# -------------------------------------------------------------------
# Public assistant chat API
# -------------------------------------------------------------------
# -------------------------------------------------------------------
# Upload config
# -------------------------------------------------------------------
ALLOWED_EXTENSIONS = {"pdf", "ppt", "pptx", "doc", "docx", "txt"}
EVENT_IMAGE_EXTS = {"png", "jpg", "jpeg", "gif"}




def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# -------------------------------------------------------------------
# App and extensions
# -------------------------------------------------------------------
app = Flask(__name__)
app.config.from_object(Config)

app.config["UPLOAD_FOLDER"] = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
EVENT_UPLOAD_FOLDER = os.path.join(app.config["UPLOAD_FOLDER"], "events")
os.makedirs(EVENT_UPLOAD_FOLDER, exist_ok=True)

Talisman(app, content_security_policy=None)

db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
)

# -------------------------------------------------------------------
# Public home
# -------------------------------------------------------------------


# -------------------------------------------------------------------
# Redirect root to public home
# -------------------------------------------------------------------
@app.route("/home")
def public_home():
    events = ClubEvent.query.filter_by(status="approved").order_by(ClubEvent.date).all()
    return render_template("public_home.html", events=events)

    

@app.route("/go-home")
def go_home():
    return redirect(url_for("public_home"))


# -------------------------------------------------------------------
# Public AI assistant chat API
# -------------------------------------------------------------------

@app.route("/api/assistant/chat", methods=["POST"])
def assistant_chat():
    data = request.get_json(silent=True) or {}
    user_message = (data.get("message") or "").strip()
    if not user_message:
        return jsonify({"reply": "Please type a question so I can help."}), 400

    # Load institution data (RAG context)
    dept_text = load_departments_knowledge()

    base_instruction = (
        "You are the public website assistant for this institution. "
        "You must answer ONLY using the information provided below as context. "
        "If the answer is not in the context, say you do not know, "
        "and suggest contacting the institution office.\n\n"
    )

    context_block = "=== Institution Departments ===\n" + dept_text + "\n=== End of context ===\n\n"

    prompt = (
        base_instruction
        + context_block
        + f"Visitor question: {user_message}\n\n"
        + "Answer in 2–4 sentences, referring only to the context above."
    )

    try:
        resp = llm.invoke(prompt)
        # resp.content is a list of message parts; resp.content[0].text usually has the text
        if hasattr(resp, "content"):
            if isinstance(resp.content, list) and resp.content:
                reply_text = getattr(resp.content[0], "text", "") or ""
            else:
                reply_text = str(resp.content)
        else:
            reply_text = str(resp)
        reply_text = reply_text.strip()
        if not reply_text:
            reply_text = (
                "I am not sure how to answer that from the available data. "
                "Please contact the institution office for details."
            )
    except Exception as e:
        print("assistant_chat ERROR:", e)
        reply_text = (
            "There was a problem contacting the assistant. "
            "Please try again later or contact the institution office."
        )


    return jsonify({"reply": reply_text})


# -------------------------------------------------------------------
# Load departments knowledge from CSV
# -------------------------------------------------------------------
import csv

DEPT_KNOWLEDGE_CSV = os.path.join(DATA_DIR, "knowledge", "departments.csv")

def load_departments_knowledge() -> str:
    """Return a short text summary of departments from departments.csv."""
    if not os.path.exists(DEPT_KNOWLEDGE_CSV):
        return ""
    lines = []
    with open(DEPT_KNOWLEDGE_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dept = (row.get("Department") or "").strip()
            desc = (row.get("Description") or "").strip()
            email = (row.get("ContactEmail") or "").strip()
            if dept:
                line = f"Department: {dept}. Description: {desc}. Contact: {email}."
                lines.append(line)
    return "\n".join(lines)

# -------------------------------------------------------------------
# Models
# -------------------------------------------------------------------
class Department(db.Model):
    __tablename__ = "departments"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)



class ClubEvent(db.Model):
    __tablename__ = "club_events"
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=False)
    date = db.Column(db.Date, nullable=True)
    venue = db.Column(db.String(200), nullable=True)
    image_name = db.Column(db.String(255))          # new: stored file name
    created_by = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    status = db.Column(db.String(20), nullable=False, default="pending")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    creator = db.relationship("User")



class User(UserMixin, db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    reg_no = db.Column(db.String(120), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=True)
    password_hash = db.Column(db.String(255), nullable=False)
    # roles: student, staff, admin, hod, examiner, club
    role = db.Column(db.String(20), nullable=False)
    full_name = db.Column(db.String(150), nullable=True)
    department_id = db.Column(db.Integer, db.ForeignKey("departments.id"))

    def set_password(self, password: str):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)


class Course(db.Model):
    __tablename__ = "courses"
    id = db.Column(db.Integer, primary_key=True)
    code = db.Column(db.String(50), unique=True, nullable=False)
    name = db.Column(db.String(200), nullable=False)
    semester = db.Column(db.String(20))
    section = db.Column(db.String(20))
    department_id = db.Column(db.Integer, db.ForeignKey("departments.id"))


class Question(db.Model):
    __tablename__ = "questions"
    id = db.Column(db.Integer, primary_key=True)
    assessment_id = db.Column(db.Integer, db.ForeignKey("assessments.id"), nullable=False)
    text = db.Column(db.Text, nullable=False)
    option_a = db.Column(db.String(255), nullable=False)
    option_b = db.Column(db.String(255), nullable=False)
    option_c = db.Column(db.String(255), nullable=False)
    option_d = db.Column(db.String(255), nullable=False)
    correct_option = db.Column(db.String(1), nullable=False)  # "A", "B", "C", or "D"
    marks = db.Column(db.Integer, nullable=False, default=1)

    assessment = db.relationship("Assessment", backref="questions")


class StudentAnswer(db.Model):
    __tablename__ = "student_answers"
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    assessment_id = db.Column(db.Integer, db.ForeignKey("assessments.id"), nullable=False)
    question_id = db.Column(db.Integer, db.ForeignKey("questions.id"), nullable=False)
    chosen_option = db.Column(db.String(1), nullable=False)  # "A","B","C","D"
    is_correct = db.Column(db.Boolean, nullable=False)

    student = db.relationship("User")
    assessment = db.relationship("Assessment")
    question = db.relationship("Question")


class StaffCourse(db.Model):
    __tablename__ = "staff_courses"
    id = db.Column(db.Integer, primary_key=True)
    staff_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    course_id = db.Column(db.Integer, db.ForeignKey("courses.id"), nullable=False)

 # already imported somewhere near top

# already imported somewhere near top

class Assessment(db.Model):
    __tablename__ = "assessments"

    id = db.Column(db.Integer, primary_key=True)
    course_id = db.Column(db.Integer, db.ForeignKey("courses.id"), nullable=False)
    title = db.Column(db.String(200), nullable=False)
    max_marks = db.Column(db.Integer, nullable=False)
    date = db.Column(db.Date)              # existing
    start_date = db.Column(db.Date)        # new
    end_date = db.Column(db.Date)          # new
    start_time = db.Column(db.DateTime)    # keep but you can ignore it
    end_time = db.Column(db.DateTime)
    created_by = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)


class AttClass(db.Model):
    """
    Attendance class, same as friend's `classes` table.
    Kept separate from your `courses`.
    """
    __tablename__ = "att_classes"

    id = db.Column("class_id", db.Integer, primary_key=True)
    class_name = db.Column(db.String(100), nullable=False)
    department = db.Column(db.String(100))


class AttStudent(db.Model):
    """
    Students used for attendance only.
    Friend's `students` table but stored in MySQL and linked if needed.
    """
    __tablename__ = "att_students"

    reg_no = db.Column(db.String(50), primary_key=True)
    student_name = db.Column(db.String(150), nullable=False)
    class_id = db.Column(db.Integer, db.ForeignKey("att_classes.class_id"), nullable=False)


class AttPeriod(db.Model):
    """
    Each class hour.
    """
    __tablename__ = "att_periods"

    id = db.Column("period_id", db.Integer, primary_key=True, autoincrement=True)
    class_id = db.Column(db.Integer, db.ForeignKey("att_classes.class_id"))
    subject_name = db.Column(db.String(150))
    period_date = db.Column(db.String(20))    # keep as text like friend's code
    period_number = db.Column(db.Integer)


class AttRecord(db.Model):
    """
    One row per student per period.
    """
    __tablename__ = "att_records"

    period_id = db.Column(db.Integer, db.ForeignKey("att_periods.period_id"), primary_key=True)
    reg_no = db.Column(db.String(50), db.ForeignKey("att_students.reg_no"), primary_key=True)
    is_present = db.Column(db.Integer, nullable=False)  # 1 or 0






class AssessmentMark(db.Model):
    __tablename__ = "assessment_marks"
    id = db.Column(db.Integer, primary_key=True)
    assessment_id = db.Column(db.Integer, db.ForeignKey("assessments.id"), nullable=False)
    student_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    marks_obtained = db.Column(db.Float, nullable=False)

    assessment = db.relationship("Assessment", backref="marks")
    student = db.relationship("User")



# -------------------------------------------------------------------
# Forms
# -------------------------------------------------------------------
class LoginForm(FlaskForm):
    reg_no = StringField("Reg No or Email", validators=[InputRequired(), Length(min=1, max=120)])
    password = PasswordField("Password", validators=[InputRequired(), Length(min=6, max=128)])
    submit = SubmitField("Login")


# -------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


def role_required(expected_role):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not current_user.is_authenticated:
                return login_manager.unauthorized()
            if current_user.role != expected_role:
                abort(403)
            return func(*args, **kwargs)
        return wrapper
    return decorator


# -------------------------------------------------------------------
# Routes: index / login / logout
# -------------------------------------------------------------------
@app.route("/")
def index():
    if current_user.is_authenticated:
        if current_user.role == "student":
            return redirect(url_for("student_home"))
        if current_user.role == "staff":
            return redirect(url_for("staff_home"))
        if current_user.role == "admin":
            return redirect(url_for("admin_home"))
        if current_user.role == "hod":
            return redirect(url_for("hod_home"))
        if current_user.role == "examiner":
            return redirect(url_for("examiner_home"))
        if current_user.role == "club":
            return redirect(url_for("club_home"))
    return redirect(url_for("public_home"))

# -------------------------------------------------------------------
# Login and logout
# -------------------------------------------------------------------
@app.route("/login", methods=["GET", "POST"])
@limiter.limit("5 per minute", override_defaults=False)
def login():
    form = LoginForm()
    if form.validate_on_submit():
        reg_no = form.reg_no.data.strip()
        password = form.password.data

        user = User.query.filter(
            (User.reg_no == reg_no) | (User.email == reg_no)
        ).first()
        if user and user.check_password(password):
            login_user(user)
            flash("Login successful", "success")
            return redirect(url_for("index"))
        else:
            flash("Invalid credentials", "danger")

    return render_template("login.html", form=form)


@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Logged out", "info")
    return redirect(url_for("login"))

# -------------------------------------------------------------------
# Student: take assessment (online test)
# -------------------------------------------------------------------





#-------------------------------------------------------------------
# Utility: check if student has attempted assessment
#-------------------------------------------------------------------

def has_attempted_assessment(student_id, assessment_id):
    return (
        StudentAnswer.query
        .filter_by(student_id=student_id, assessment_id=assessment_id)
        .first()
        is not None
    )


# -------------------------------------------------------------------
# Student: my courses
# -------------------------------------------------------------------

@app.route("/student/courses")
@login_required
@role_required("student")
def student_courses():
    courses = Course.query.all()  # or filter by this student's courses
    return render_template("student/student_courses.html", courses=courses)

# -------------------------------------------------------------------
# Student: assessments list for a course
# -------------------------------------------------------------------


@app.route("/student/courses/<int:course_id>/assessments")
@login_required
@role_required("student")
def student_assessments(course_id):
    course = Course.query.get_or_404(course_id)

    assessments = (
        Assessment.query
        .filter_by(course_id=course.id)
        .order_by(Assessment.date.desc())
        .all()
    )

    attempted_ids = {
        row.assessment_id
        for row in (
            StudentAnswer.query
            .with_entities(StudentAnswer.assessment_id)
            .filter_by(student_id=current_user.id)
            .distinct()
            .all()
        )
    }

    return render_template(
        "student/student_assessments.html",
        course=course,
        assessments=assessments,
        attempted_ids=attempted_ids,
    )


# -------------------------------------------------------------------
# Student: view all marks
# -------------------------------------------------------------------

@app.route("/student/marks")
@login_required
@role_required("student")
def student_marks():
    rows = (
        db.session.query(
            Assessment.id,
            Assessment.title,
            Assessment.max_marks,
            Course.code,
            Course.name,
            func.sum(
                case(
                    (StudentAnswer.is_correct == True, Question.marks),
                    else_=0
                )
            ).label("score")
        )
        .join(StudentAnswer, StudentAnswer.assessment_id == Assessment.id)
        .join(Question, StudentAnswer.question_id == Question.id)
        .join(Course, Assessment.course_id == Course.id)
        .filter(StudentAnswer.student_id == current_user.id)
        .group_by(
            Assessment.id,
            Assessment.title,
            Assessment.max_marks,
            Course.code,
            Course.name,
        )
        .all()
    )

    return render_template("student/student_marks.html", rows=rows)


# -------------------------------------------------------------------
# Student dashboards
# -------------------------------------------------------------------
@app.route("/student/home")
@login_required
@role_required("student")
def student_home():
    my_courses = [
        {"code": "CSE201", "name": "Data Structures"},
        {"code": "CSE305", "name": "Operating Systems"},
    ]
    upcoming_assessments = [
        {"course": "CSE201", "title": "Unit Test 1", "date": "2025-12-20"},
        {"course": "CSE305", "title": "Lab Internal", "date": "2025-12-22"},
    ]
    attendance = [
        {"course": "CSE201", "percent": 82},
        {"course": "CSE305", "percent": 68},
    ]
    return render_template(
        "student/student_dashboard.html",
        my_courses=my_courses,
        upcoming_assessments=upcoming_assessments,
        attendance=attendance,
    )

#-----------------------------------------------------------------------------------------------------------------
#
#------------------------------------------------------------------------------------------------------------------

@app.route("/student/materials")
@login_required
@role_required("student")
def student_materials():
    result = db.session.execute(
        db.text(
            "SELECT c.id AS course_id, c.code, c.name "
            "FROM enrollments e "
            "JOIN courses c ON e.course_id = c.id "
            "WHERE e.student_id = :sid"
        ),
        {"sid": current_user.id},
    )
    courses = result.mappings().all()

    course_ids = [c["course_id"] for c in courses]
    files_by_course = {cid: [] for cid in course_ids}
    if course_ids:
        rows = db.session.execute(
            db.text(
                "SELECT cf.course_id, cf.original_name, cf.stored_name, cf.uploaded_at "
                "FROM course_files cf "
                "WHERE cf.course_id IN :ids "
                "ORDER BY cf.uploaded_at DESC"
            ),
            {"ids": tuple(course_ids)},
        ).mappings().all()
        for r in rows:
            files_by_course[r["course_id"]].append(r)

    return render_template(
        "student/student_materials.html",
        courses=courses,
        files_by_course=files_by_course,
    )


# -------------------------------------------------------------------
# Staff dashboards
# -------------------------------------------------------------------
@app.route("/staff/home")
@login_required
@role_required("staff")
def staff_home():
    stats = {
        "total_courses": 3,
        "upcoming_exams": 2,
        "pending_submissions": 15,
    }
    return render_template("staff/staff_dashboard.html", stats=stats)


@app.route("/staff/dashboard")
@login_required
@role_required("staff")
def staff_dashboard():
    staff_id = current_user.id
    courses = (
        db.session.query(Course)
        .join(StaffCourse, StaffCourse.course_id == Course.id)
        .filter(StaffCourse.staff_id == staff_id)
        .all()
    )
    return render_template("staff/staff_dashboard.html", courses=courses)

#-------------------------------------------------------------------
# Staff: upload assessment marks
#-------------------------------------------------------------------

@app.route("/staff/assessments/<int:assessment_id>/upload-marks", methods=["GET", "POST"])
@login_required
@role_required("staff")
def staff_upload_marks(assessment_id):
    assessment = Assessment.query.get_or_404(assessment_id)
    # security: confirm this staff teaches this course
    link = StaffCourse.query.filter_by(staff_id=current_user.id, course_id=assessment.course_id).first()
    if not link:
        abort(403)

    if request.method == "POST":
        file = request.files.get("file")
        if not file or not file.filename:
            flash("No file selected.", "danger")
            return redirect(request.url)

        import csv, io
        text = file.read().decode("utf-8")
        reader = csv.DictReader(io.StringIO(text))
        # expected header: RegNo, Marks
        count = 0
        from sqlalchemy import and_

        for row in reader:
            regno = (row.get("RegNo") or "").strip()
            marks_raw = (row.get("Marks") or "").strip()
            if not regno or not marks_raw:
                continue
            try:
                marks = float(marks_raw)
            except Exception:
                continue
            student = User.query.filter_by(reg_no=regno, role="student").first()
            if not student:
                continue
            # upsert
            rec = AssessmentMark.query.filter(
                and_(
                    AssessmentMark.assessment_id == assessment.id,
                    AssessmentMark.student_id == student.id,
                )
            ).first()
            if rec:
                rec.marks_obtained = marks
            else:
                rec = AssessmentMark(
                    assessment_id=assessment.id,
                    student_id=student.id,
                    marks_obtained=marks,
                )
                db.session.add(rec)
            count += 1

        db.session.commit()
        flash(f"Uploaded/updated marks for {count} students.", "success")
        return redirect(url_for("staff_assessments", course_id=assessment.course_id))

    return render_template("staff/staff_upload_marks.html", assessment=assessment)


def get_scores_for_assessment(assessment_id):
    rows = (
        db.session.query(
            StudentAnswer.student_id,
            func.sum(
                case(
                    (StudentAnswer.is_correct == True, Question.marks),
                    else_=0
                )
            ).label("score")
        )
        .join(Question, StudentAnswer.question_id == Question.id)
        .filter(StudentAnswer.assessment_id == assessment_id)
        .group_by(StudentAnswer.student_id)
        .all()
    )
    return rows




#-------------------------------------------------------------------
# Student: take assessment (placeholder)
#-------------------------------------------------------------------




#-------------------------------------------------------------------
# Student: start assessment (check time window)
#-------------------------------------------------------------------



@app.route("/student/assessments/<int:assessment_id>/start", methods=["GET", "POST"])
@login_required
@role_required("student")
def student_start_assessment(assessment_id):
    assessment = Assessment.query.get_or_404(assessment_id)

    # date window check
    today = date.today()
    if assessment.start_date and today < assessment.start_date:
        flash("You are not yet eligible to take this test (start date not reached).", "warning")
        return redirect(url_for("student_assessments", course_id=assessment.course_id))

    if assessment.end_date and today > assessment.end_date:
        flash("You are not eligible to take this test (end date is over).", "warning")
        return redirect(url_for("student_assessments", course_id=assessment.course_id))

    # already attempted?
    if has_attempted_assessment(current_user.id, assessment.id):
        flash("You have already submitted this test.", "warning")
        return redirect(url_for("student_assessments", course_id=assessment.course_id))

    # load questions
    questions = Question.query.filter_by(assessment_id=assessment.id).all()

    if request.method == "POST":
        total = 0
        for q in questions:
            chosen = request.form.get(f"q_{q.id}")
            if not chosen:
                continue
            is_correct = (chosen == q.correct_option)
            if is_correct:
                total += q.marks
            ans = StudentAnswer(
                student_id=current_user.id,
                assessment_id=assessment.id,
                question_id=q.id,
                chosen_option=chosen,
                is_correct=is_correct,
            )
            db.session.add(ans)

        db.session.commit()
        flash(f"Test submitted. Your score: {total} / {assessment.max_marks}", "success")
        return redirect(url_for("student_assessments", course_id=assessment.course_id))

    return render_template(
        "student/student_take_assessment.html",
        assessment=assessment,
        questions=questions,
    )


#-------------------------------------------------------------------
# Student: view all assessments
#-------------------------------------------------------------------
@app.route("/student/assessments")
@login_required
@role_required("student")
def student_assessments_list():
    # all assessments for courses this student is enrolled in
    rows = db.session.execute(
        db.text(
            """
            SELECT a.id, a.title, a.max_marks, a.date,
                   a.start_time, a.end_time,
                   c.code AS course_code, c.name AS course_name
            FROM assessments a
            JOIN courses c ON a.course_id = c.id
            JOIN enrollments e ON e.course_id = c.id
            WHERE e.student_id = :sid
            ORDER BY a.date DESC
            """
        ),
        {"sid": current_user.id},
    ).mappings().all()

    return render_template("student/student_assessments.html", rows=rows)


#-------------------------------------------------------------------
# Staff: view submissions for an assessment
#-------------------------------------------------------------------

@app.route("/staff/assessments/<int:assessment_id>/submissions")
@login_required
@role_required("staff")
def staff_view_submissions(assessment_id):
    assessment = Assessment.query.get_or_404(assessment_id)

    # ensure this staff owns the course
    link = StaffCourse.query.filter_by(
        staff_id=current_user.id,
        course_id=assessment.course_id,
    ).first()
    if not link:
        abort(403)

    scores = get_scores_for_assessment(assessment.id)
    # map student_id -> score
    score_map = {sid: sc for sid, sc in scores}

    # fetch student objects for display
    student_ids = list(score_map.keys())
    students = User.query.filter(User.id.in_(student_ids)).all()

    return render_template(
        "staff/staff_assessment_submissions.html",
        assessment=assessment,
        students=students,
        score_map=score_map,
    )


#-------------------------------------------------------------------
# Staff: manage questions for an assessment
#-------------------------------------------------------------------


@app.route("/staff/assessments/<int:assessment_id>/questions", methods=["GET", "POST"])
@login_required
@role_required("staff")
def staff_manage_questions(assessment_id):
    assessment = Assessment.query.get_or_404(assessment_id)
    # optional: check staff owns this course via StaffCourse
    link = StaffCourse.query.filter_by(
        staff_id=current_user.id,
        course_id=assessment.course_id
    ).first()
    if not link:
        abort(403)

    if request.method == "POST":
        text = request.form.get("text", "").strip()
        option_a = request.form.get("option_a", "").strip()
        option_b = request.form.get("option_b", "").strip()
        option_c = request.form.get("option_c", "").strip()
        option_d = request.form.get("option_d", "").strip()
        correct_option = request.form.get("correct_option", "").strip().upper()
        marks = request.form.get("marks", "").strip()

        if not text or not option_a or not option_b or not option_c or not option_d:
            flash("All question and option fields are required.", "danger")
        elif correct_option not in {"A", "B", "C", "D"}:
            flash("Correct option must be A, B, C, or D.", "danger")
        else:
            try:
                m = int(marks or 1)
            except ValueError:
                flash("Marks must be a number.", "danger")
            else:
                q = Question(
                    assessment_id=assessment.id,
                    text=text,
                    option_a=option_a,
                    option_b=option_b,
                    option_c=option_c,
                    option_d=option_d,
                    correct_option=correct_option,
                    marks=m,
                )
                db.session.add(q)
                db.session.commit()
                flash("Question added.", "success")

        return redirect(url_for("staff_manage_questions", assessment_id=assessment.id))

    questions = Question.query.filter_by(assessment_id=assessment.id).all()
    return render_template(
        "staff/staff_manage_questions.html",
        assessment=assessment,
        questions=questions,
    )


#-------------------------------------------------------------------
# Staff: assessments list + create
#-------------------------------------------------------------------
from datetime import datetime, date
from flask import render_template, request, redirect, url_for, flash
from flask_login import login_required, current_user

# make sure Assessment and Course are already imported / defined above

from datetime import datetime, date
from flask import render_template, request, redirect, url_for, flash
from flask_login import login_required, current_user

# Staff assessments: list + create
from datetime import datetime, date
from flask import render_template, request, redirect, url_for, flash
from flask_login import login_required, current_user

# Staff assessments: list + create
from datetime import datetime, date
from flask import render_template, request, redirect, url_for, flash
from flask_login import login_required, current_user

@app.route("/staff/courses/<int:course_id>/assessments", methods=["GET", "POST"])
@login_required
def staff_assessments(course_id):
    # only staff can access
    if current_user.role != "staff":
        flash("Access denied.", "danger")
        return redirect(url_for("index"))

    course = Course.query.get_or_404(course_id)

    if request.method == "POST":
        title = request.form.get("title", "").strip()
        max_marks = request.form.get("max_marks", "").strip()
        date_str = request.form.get("date", "").strip()
        start_date_str = request.form.get("start_date", "").strip()
        end_date_str = request.form.get("end_date", "").strip()

        if not title or not max_marks:
            flash("Title and max marks are required.", "danger")
        else:
            try:
                max_m = int(max_marks)
            except ValueError:
                flash("Max marks must be a number.", "danger")
            else:
                main_date = datetime.strptime(date_str, "%Y-%m-%d").date() if date_str else None
                start_date_val = datetime.strptime(start_date_str, "%Y-%m-%d").date() if start_date_str else None
                end_date_val = datetime.strptime(end_date_str, "%Y-%m-%d").date() if end_date_str else None

                new_assessment = Assessment(
                    course_id=course.id,
                    title=title,
                    max_marks=max_m,
                    date=main_date,
                    start_date=start_date_val,
                    end_date=end_date_val,
                    start_time=None,
                    end_time=None,
                    created_by=current_user.id,
                )
                db.session.add(new_assessment)
                db.session.commit()
                flash("Assessment created.", "success")

        return redirect(url_for("staff_assessments", course_id=course.id))

    # GET: list all assessments for this course
    assessments = Assessment.query.filter_by(course_id=course.id).order_by(Assessment.id.desc()).all()
    return render_template("staff/staff_assessments.html", course=course, assessments=assessments)



#-------------------------------------------------------------------
# Staff: my courses + upload materials
#-------------------------------------------------------------------


@app.route("/staff/courses")
@login_required
@role_required("staff")
def staff_my_courses():
    staff_id = current_user.id
    print("DEBUG staff_my_courses current_user:", staff_id, current_user.reg_no)
    courses = (
        db.session.query(Course)
        .join(StaffCourse, StaffCourse.course_id == Course.id)
        .filter(StaffCourse.staff_id == staff_id)
        .all()
    )
    return render_template("staff/staff_courses.html", courses=courses)


#-------------------------------------------------------------------
# Staff: course materials upload + list
#-------------------------------------------------------------------


@app.route("/staff/courses/<int:course_id>/materials", methods=["GET", "POST"])
@login_required
@role_required("staff")
def staff_course_materials(course_id):
    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            flash("No file selected.", "danger")
            return redirect(request.url)

        if not allowed_file(file.filename):
            flash("File type not allowed.", "danger")
            return redirect(request.url)

        original_name = file.filename
        safe_name = secure_filename(original_name)
        stored_name = f"{course_id}_{current_user.id}_{safe_name}"
        file.save(os.path.join(app.config["UPLOAD_FOLDER"], stored_name))

        db.session.execute(
            db.text(
                "INSERT INTO course_files (course_id, original_name, stored_name, uploaded_by) "
                "VALUES (:course_id, :original_name, :stored_name, :uploaded_by)"
            ),
            {
                "course_id": course_id,
                "original_name": original_name,
                "stored_name": stored_name,
                "uploaded_by": current_user.id,
            },
        )
        db.session.commit()
        flash("File uploaded successfully.", "success")
        return redirect(request.url)

    result = db.session.execute(
        db.text(
            "SELECT id, original_name, stored_name, uploaded_at "
            "FROM course_files WHERE course_id = :course_id "
            "ORDER BY uploaded_at DESC"
        ),
        {"course_id": course_id},
    )
    files = result.mappings().all()

    return render_template(
        "staff/staff_course_materials.html",
        course_id=course_id,
        files=files,
    )


@app.route("/materials/<path:stored_name>")
@login_required
def download_material(stored_name):
    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        stored_name,
        as_attachment=True,
    )

#-------------------------------------------------------------------
# Staff: attendance and exam schedule
#-------------------------------------------------------------------


#-------------------------------------------------------------------
# Staff: exam schedule
#-------------------------------------------------------------------

@app.route("/staff/exams")
@login_required
@role_required("staff")
def staff_exam_schedule():
    exams = [
        {"course": "CSE201", "date": "2025-12-20", "time": "10:00–12:00", "room": "Block B – 205"},
        {"course": "CSE305", "date": "2025-12-22", "time": "14:00–16:00", "room": "Block C – 101"},
    ]
    return render_template("staff/staff_exam_schedule.html", exams=exams)




# ---------- Staff: mark attendance ----------

@app.route("/staff/attendance", methods=["GET"])
@login_required
@role_required("staff")
def staff_attendance():
    return render_template("attendance/staff.html")


@app.route("/api/att/periods", methods=["POST"])
@login_required
@role_required("staff")
def api_att_create_period():
    data = request.json or {}
    p = AttPeriod(
        class_id=data["class_id"],
        subject_name=data["subject_name"],
        period_date=data["period_date"],
        period_number=data["period_number"],
    )
    db.session.add(p)
    db.session.commit()
    return jsonify({"message": "Period created", "period_id": p.id}), 201


@app.route("/api/att/attendance", methods=["POST"])
@login_required
@role_required("staff")
def api_att_mark():
    data = request.json or {}
    period_id = data["period_id"]
    items = data["attendance"]

    for rec in items:
        row = AttRecord(
            period_id=period_id,
            reg_no=rec["reg_no"],
            is_present=int(rec["is_present"]),
        )
        # INSERT OR REPLACE: emulate with merge semantics
        existing = AttRecord.query.filter_by(period_id=period_id, reg_no=row.reg_no).first()
        if existing:
            existing.is_present = row.is_present
        else:
            db.session.add(row)
    db.session.commit()
    return jsonify({"message": "Attendance saved"}), 200


@app.route("/api/att/attendance/<reg_no>", methods=["GET"])
@login_required
def api_att_overall(reg_no):
    # any logged-in user can query; add role checks if needed
    from sqlalchemy import func, case

    q = (
        db.session.query(
            AttStudent.reg_no,
            AttStudent.student_name,
            func.count(AttRecord.period_id),
            func.sum(case((AttRecord.is_present == 1, 1), else_=0)),
        )
        .outerjoin(AttRecord, AttStudent.reg_no == AttRecord.reg_no)
        .filter(AttStudent.reg_no == reg_no)
        .group_by(AttStudent.reg_no, AttStudent.student_name)
    )
    row = q.first()
    if not row:
        return jsonify({"error": "Student not found"}), 404

    total = row[2] or 0
    attended = row[3] or 0
    pct = round(attended * 100.0 / total, 2) if total else 0.0

    return jsonify({
        "reg_no": row[0],
        "student_name": row[1],
        "total_classes": total,
        "attended_classes": attended,
        "attendance_percentage": pct,
    })

# -------------------------------------------------------------------
# Admin home + course upload (principal level)
# -------------------------------------------------------------------
@app.route("/admin/home")
@login_required
@role_required("admin")
def admin_home():
    # HOD list for dropdown
    hods = User.query.filter_by(role="hod").all()

    # all courses
    courses = Course.query.order_by(Course.semester, Course.code).all()

    # simple stats for dashboard cards
    total_hods = len(hods)
    total_staff = User.query.filter_by(role="staff").count()
    total_students = User.query.filter_by(role="student").count()
    total_courses = len(courses)

    return render_template(
        "admin/admin_dashboard.html",
        hods=hods,
        courses=courses,
        total_hods=total_hods,
        total_staff=total_staff,
        total_students=total_students,
        total_courses=total_courses,
    )


# ---------- Admin: attendance management ----------

@app.route("/admin/attendance")
@login_required
@role_required("admin")
def admin_attendance_home():
    return render_template("attendance/admin.html")


# API: classes
@app.route("/api/att/classes", methods=["POST"])
@login_required
@role_required("admin")
def api_att_add_class():
    data = request.json or {}
    cls = AttClass(
        id=data["class_id"],
        class_name=data["class_name"],
        department=data.get("department", ""),
    )
    db.session.add(cls)
    try:
        db.session.commit()
        return jsonify({"message": "Class added"}), 201
    except Exception:
        db.session.rollback()
        return jsonify({"error": "Class ID already exists"}), 400


@app.route("/api/att/classes", methods=["GET"])
@login_required
@role_required("admin")
def api_att_get_classes():
    rows = AttClass.query.all()
    return jsonify([
        {"class_id": r.id, "class_name": r.class_name, "department": r.department}
        for r in rows
    ])


# API: students (single add)
@app.route("/api/att/students", methods=["POST"])
@login_required
@role_required("admin")
def api_att_add_student():
    data = request.json or {}
    stu = AttStudent(
        reg_no=data["reg_no"],
        student_name=data["student_name"],
        class_id=data["class_id"],
    )
    db.session.add(stu)
    try:
        db.session.commit()
        return jsonify({"message": "Student added"}), 201
    except Exception:
        db.session.rollback()
        return jsonify({"error": "Reg no already exists"}), 400


@app.route("/api/att/students", methods=["GET"])
@login_required
@role_required("admin")
def api_att_get_students():
    class_id = request.args.get("class_id")
    q = AttStudent.query
    if class_id:
        q = q.filter_by(class_id=class_id)
    rows = q.all()
    return jsonify([
        {"reg_no": r.reg_no, "student_name": r.student_name, "class_id": r.class_id}
        for r in rows
    ])


# API: bulk CSV upload (admin)
@app.route("/api/att/students/bulk", methods=["POST"])
@login_required
@role_required("admin")
def api_att_bulk_students():
    import csv, io
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    stream = io.StringIO(file.stream.read().decode("utf-8"), newline=None)
    reader = csv.reader(stream)
    next(reader, None)  # skip header

    added = 0
    skipped = 0
    for row in reader:
        if len(row) < 3:
            continue
        try:
            stu = AttStudent(
                reg_no=row[0].strip(),
                student_name=row[1].strip(),
                class_id=int(row[2].strip()),
            )
            db.session.add(stu)
            db.session.flush()
            added += 1
        except Exception:
            db.session.rollback()
            skipped += 1
    db.session.commit()
    return jsonify({
        "message": f"{added} students added, {skipped} skipped"
    }), 201

# -------------------------------------------------------------------
# Admin: manage club events
# -------------------------------------------------------------------   

@app.route("/admin/events", methods=["GET", "POST"])
@login_required
@role_required("admin")
def admin_events():
    if request.method == "POST":
        event_id = int(request.form.get("event_id", 0))
        action = request.form.get("action")
        ev = ClubEvent.query.get_or_404(event_id)
        if action == "approve":
            ev.status = "approved"
        elif action == "reject":
            ev.status = "rejected"
        db.session.commit()
        flash("Event status updated.", "success")
        return redirect(url_for("admin_events"))

    events = ClubEvent.query.order_by(ClubEvent.created_at.desc()).all()
    return render_template("admin/admin_events.html", events=events)

# -------------------------------------------------------------------
# Admin assign courses to HOD / department
# -------------------------------------------------------------------

@app.route("/admin/assign-courses", methods=["POST"])
@login_required
@role_required("admin")
def admin_assign_courses():
    hod_id = int(request.form["hod_id"])
    hod = User.query.get_or_404(hod_id)

    course_ids = [int(cid) for cid in request.form.getlist("course_ids")]
    assign_mode = request.form.get("assign_mode", "add")

    if not course_ids:
        flash("No courses selected.", "warning")
        return redirect(url_for("admin_home"))

    if not hod.department_id:
        flash("Selected HOD has no department set.", "danger")
        return redirect(url_for("admin_home"))

    if assign_mode == "replace":
        old_courses = Course.query.filter_by(department_id=hod.department_id).all()
        for c in old_courses:
            c.department_id = None

    for cid in course_ids:
        course = Course.query.get(cid)
        if course:
            course.department_id = hod.department_id

    db.session.commit()

    if assign_mode == "replace":
        flash("Department courses replaced for this HOD.", "success")
    else:
        flash("Courses added to this HOD's department.", "success")

    return redirect(url_for("admin_home"))


#---------------------------------------------------------------------------
#
#---------------------------------------------------------------------------------

@app.route("/admin/upload-courses", methods=["GET", "POST"])
@login_required
@role_required("admin")
def upload_courses():
    if request.method == "POST":
        file = request.files["file"]
        if not file or file.filename == "":
            flash("No file selected", "danger")
            return redirect(request.url)

        

        if file.filename.lower().endswith((".xls", ".xlsx")):
            
            df = pd.read_excel(file)
        else:
            df = pd.read_csv(file)

        df.columns = [c.strip() for c in df.columns]

        # expected columns: Code, Name, Semester, Section, Department
        for _, row in df.iterrows():
            code = str(row["Code"]).strip()
            name = str(row["Name"]).strip()
            sem = str(row["Semester"]).strip()
            sec = str(row["Section"]).strip()
            dept_name = str(row["Department"]).strip()

            dept = Department.query.filter_by(name=dept_name).first()
            if not dept:
                dept = Department(name=dept_name)
                db.session.add(dept)
                db.session.flush()

            existing = Course.query.filter_by(code=code).first()
            if existing:
                existing.name = name
                existing.semester = sem
                existing.section = sec
                existing.department_id = dept.id
            else:
                course = Course(
                    code=code,
                    name=name,
                    semester=sem,
                    section=sec,
                    department_id=dept.id,
                )
                db.session.add(course)

        db.session.commit()
        flash("Courses imported/updated from file", "success")
        return redirect(url_for("admin_home"))

    return render_template("admin/upload_courses.html")

#---------------------------------------------------------------------------
# Admin upload staff list
#----------------------------------------------------------------------------


@app.route("/admin/upload-staff", methods=["GET", "POST"])
@login_required
@role_required("admin")
def admin_upload_staff():
    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            flash("No file selected", "danger")
            return redirect(url_for("admin_upload_staff"))

        import pandas as pd
        if file.filename.lower().endswith((".xls", ".xlsx")):
            import openpyxl
            df = pd.read_excel(file)
        else:
            df = pd.read_csv(file)

        df.columns = [c.strip() for c in df.columns]

        for _, row in df.iterrows():
            reg_no = str(row["RegNo"]).strip()
            full_name = str(row["FullName"]).strip()
            email = str(row["Email"]).strip()

            existing = User.query.filter_by(reg_no=reg_no).first()
            if existing:
                continue

            u = User(
                reg_no=reg_no,
                full_name=full_name,
                email=email,
                role="staff",
            )
            u.set_password("password123")
            db.session.add(u)

        db.session.commit()
        flash("Staff list uploaded.", "success")
        return redirect(url_for("admin_home"))

    return render_template("admin/admin_upload_staff.html")

#---------------------------------------------------------------------------------------------
#upload hods
#---------------------------------------------------------------------------------------------

@app.route("/admin/upload-hods", methods=["GET", "POST"])
@login_required
@role_required("admin")
def admin_upload_hods():
    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            flash("No file selected", "danger")
            return redirect(url_for("admin_upload_hods"))

        import pandas as pd
        if file.filename.lower().endswith((".xls", ".xlsx")):
            import openpyxl
            df = pd.read_excel(file)
        else:
            df = pd.read_csv(file)

        df.columns = [c.strip() for c in df.columns]

        for _, row in df.iterrows():
            reg_no = str(row["RegNo"]).strip()
            full_name = str(row["FullName"]).strip()
            email = str(row["Email"]).strip()
            dept_name = str(row["Department"]).strip()

            # get or create department
            dept = Department.query.filter_by(name=dept_name).first()
            if not dept:
                dept = Department(name=dept_name)
                db.session.add(dept)
                db.session.flush()

            existing = User.query.filter_by(reg_no=reg_no).first()
            if existing:
                continue

            u = User(
                reg_no=reg_no,
                full_name=full_name,
                email=email,
                role="hod",
                department_id=dept.id,
            )
            u.set_password("password123")
            db.session.add(u)

        db.session.commit()
        flash("HOD list uploaded.", "success")
        return redirect(url_for("admin_home"))

    return render_template("admin/admin_upload_hods.html")




# -------------------------------------------------------------------
# Admin AI Chat
# -------------------------------------------------------------------
@app.route("/admin/ai-chat", methods=["GET", "POST"])
@login_required
@role_required("admin")
def admin_ai_chat():
    messages = []

    if request.method == "POST":
        user_text = request.form.get("message", "").strip()
        if user_text:
            chat = gemini_client.chats.create(model=GEMINI_MODEL)
            response = chat.send_message(user_text)
            bot_text = response.text
            messages.append({"role": "user", "text": user_text})
            messages.append({"role": "ai", "text": bot_text})

    return render_template("admin/admin_ai_chat.html", messages=messages)


# -------------------------------------------------------------------
# HOD: assign courses to staff (inside department)
# -------------------------------------------------------------------
@app.route("/hod/assign-courses", methods=["POST"])
@login_required
@role_required("hod")
def hod_assign_courses():
    # read values from the form
    staff_id = int(request.form["staff_id"])
    course_ids = [int(cid) for cid in request.form.getlist("course_ids")]

    if not course_ids:
        flash("No courses selected.", "warning")
        return redirect(url_for("hod_home"))

    # ensure staff is in this HOD's department
    staff_user = User.query.filter_by(
        id=staff_id,
        role="staff",
        department_id=current_user.department_id,
    ).first_or_404()

    # delete ALL previous links for this staff (simpler, ignore department)
    old_links = StaffCourse.query.filter_by(staff_id=staff_user.id).all()
    for link in old_links:
        db.session.delete(link)

    # create new links for selected courses
    for cid in course_ids:
        course = Course.query.get(cid)
        if course:
            db.session.add(StaffCourse(staff_id=staff_user.id, course_id=course.id))

    db.session.commit()
    flash("Courses assigned to staff.", "success")
    return redirect(url_for("hod_home"))


# -------------------------------------------------------------------
# HOD dashboard (department-level)
# -------------------------------------------------------------------
@app.route("/hod/home")
@login_required
@role_required("hod")
def hod_home():
    staff = User.query.filter_by(
        role="staff",
        department_id=current_user.department_id,
    ).all()

    courses = Course.query.filter_by(
        department_id=current_user.department_id,
    ).order_by(Course.semester, Course.code).all()

    return render_template("hod/hod_dashboard.html", staff=staff, courses=courses)


#-------------------------------------------------------------------
# HOD upload staff list
#-------------------------------------------------------------------

@app.route("/hod/upload-staff", methods=["GET", "POST"])
@login_required
@role_required("hod")
def hod_upload_staff():
    if request.method == "POST":
        file = request.files["file"]   # form on hod_upload_staff.html must have input name="file"
        if not file or file.filename == "":
            flash("No file selected", "danger")
            return redirect(request.url)

        import pandas as pd
        if file.filename.lower().endswith((".xls", ".xlsx")):
            import openpyxl
            df = pd.read_excel(file)
        else:
            df = pd.read_csv(file)

        df.columns = [c.strip() for c in df.columns]
        dept_id = current_user.department_id

        # expected columns: RegNo, FullName, Email
        for _, row in df.iterrows():
            reg_no = str(row["RegNo"]).strip()
            full_name = str(row["FullName"]).strip()
            email = str(row["Email"]).strip() or None

            if not reg_no:
                continue

            # try to find an existing user by reg_no or email
            existing = User.query.filter(
                (User.reg_no == reg_no) | (User.email == email)
            ).first()

            if existing:
                # update existing staff in this department
                existing.full_name = full_name or existing.full_name
                existing.email = email or existing.email
                existing.role = "staff"
                existing.department_id = dept_id
            else:
                # create new staff
                u = User(
                    reg_no=reg_no,
                    full_name=full_name,
                    email=email,
                    role="staff",
                    department_id=dept_id,
                )
                u.set_password("changeme123")
                db.session.add(u)
        db.session.commit()
        flash("Staff list imported/updated", "success")
        return redirect(url_for("hod_home"))

    return render_template("hod/hod_upload_staff.html")


# -------------------------------------------------------------------
# Examiner dashboard
# -------------------------------------------------------------------
@app.route("/examiner/home")
@login_required
@role_required("examiner")
def examiner_home():
    return render_template("examiner/examiner_dashboard.html")


# ----- Examiner hall allocation views -----

@app.route("/examiner/hall-allocation")
@login_required
@role_required("examiner")
def examiner_hall_allocation():
    # preview students file
    if os.path.exists(STUDENTS_CSV):
        with open(STUDENTS_CSV, encoding="utf-8") as f:
            preview = "".join(f.readlines()[:50])
    else:
        preview = "(no students.csv uploaded yet)"

    seating_files = sorted(
        [
            fn
            for fn in os.listdir(DATA_DIR)
            if fn.startswith("seating-") and fn.endswith(".csv")
        ],
        reverse=True,
    )

    return render_template(
        "examiner/examiner_hall_allocation.html",
        preview=preview,
        files=seating_files,
        generated_filename=None,
        mapping=[],
        preview_generated="",
    )


@app.route("/examiner/hall/upload-students", methods=["POST"])
@login_required
@role_required("examiner")
def examiner_hall_upload_students():
    file = request.files.get("students_csv")
    if not file:
        flash("No students.csv file uploaded.", "danger")
        return redirect(url_for("examiner_hall_allocation"))
    data = file.read()
    try:
        with open(STUDENTS_CSV, "wb") as f:
            f.write(data)
    except Exception as e:
        flash(f"Failed to save students.csv: {e}", "danger")
        return redirect(url_for("examiner_hall_allocation"))
    flash("students.csv uploaded successfully.", "success")
    return redirect(url_for("examiner_hall_allocation"))


@app.route("/examiner/hall/upload-classes", methods=["POST"])
@login_required
@role_required("examiner")
def examiner_hall_upload_classes():
    file = request.files.get("classes_config")
    if not file:
        flash("No classes_config file uploaded.", "danger")
        return redirect(url_for("examiner_hall_allocation"))
    data = file.read()
    try:
        with open(CLASSES_CONFIG_CSV, "wb") as f:
            f.write(data)
    except Exception as e:
        flash(f"Failed to save classes_config.csv: {e}", "danger")
        return redirect(url_for("examiner_hall_allocation"))
    flash("classes_config.csv uploaded successfully.", "success")
    return redirect(url_for("examiner_hall_allocation"))


@app.route("/examiner/hall/upload-halls", methods=["POST"])
@login_required
@role_required("examiner")
def examiner_hall_upload_halls():
    file = request.files.get("hall_config")
    if not file:
        flash("No hall_config file uploaded.", "danger")
        return redirect(url_for("examiner_hall_allocation"))
    data = file.read()
    try:
        with open(HALL_CONFIG_CSV, "wb") as f:
            f.write(data)
    except Exception as e:
        flash(f"Failed to save hall_config.csv: {e}", "danger")
        return redirect(url_for("examiner_hall_allocation"))
    flash("hall_config.csv uploaded successfully.", "success")
    return redirect(url_for("examiner_hall_allocation"))


@app.route("/examiner/hall/generate", methods=["POST"])
@login_required
@role_required("examiner")
def examiner_hall_generate():
    if not os.path.exists(STUDENTS_CSV):
        flash("No students.csv found. Upload first.", "danger")
        return redirect(url_for("examiner_hall_allocation"))

    classes_config = read_classes_config(CLASSES_CONFIG_CSV)
    halls = read_halls(HALL_CONFIG_CSV)
    if not halls:
        flash("No valid halls in hall_config.csv.", "danger")
        return redirect(url_for("examiner_hall_allocation"))

    try:
        classes = read_students_from_csv(STUDENTS_CSV)
    except Exception as e:
        flash(f"Failed to read students CSV: {e}", "danger")
        return redirect(url_for("examiner_hall_allocation"))

    from collections import deque, defaultdict
    import random
    from datetime import datetime

    class_benches: Dict[str, int] = {
        cls: 10_000 for cls in classes.keys()
    }



    for cls in classes:
        random.shuffle(classes[cls])

    remaining: Dict[str, deque] = {
        cls: deque(regs) for cls, regs in classes.items()
    }
    benches_used: Dict[str, int] = {cls: 0 for cls in classes}

    hall_index = 0
    current_hall_name, seats_per_bench, hall_bench_limit = halls[hall_index]
    hall_bench_used = 0

    all_rows: List[Tuple[str, int, int, str, str]] = []
    global_bench_number = 1

    while True:
        if hall_index >= len(halls):
            break

        placed_any = False
        bench: List[Tuple[str, str]] = []
        used_classes_in_bench = set()

        while len(bench) < seats_per_bench:
            candidate_cls = None
            for cls, queue in remaining.items():
                if not queue:
                    continue
                if benches_used[cls] >= class_benches.get(cls, 0):
                    continue
                if cls in used_classes_in_bench:
                    continue
                candidate_cls = cls
                break

            if candidate_cls is None:
                break

            reg = remaining[candidate_cls].popleft()
            bench.append((candidate_cls, reg))
            used_classes_in_bench.add(candidate_cls)
            placed_any = True

        if not placed_any:
            any_legal_left = False
            for cls, queue in remaining.items():
                if queue and benches_used[cls] < class_benches.get(cls, 0):
                    any_legal_left = True
                    break
            if not any_legal_left:
                break
            else:
                break

        if hall_bench_used >= hall_bench_limit:
            hall_index += 1
            if hall_index >= len(halls):
                break
            current_hall_name, seats_per_bench, hall_bench_limit = halls[hall_index]
            hall_bench_used = 0

        hall_bench_used += 1
        for seat_no, (cls, reg) in enumerate(bench, start=1):
            benches_used[cls] += 1
            all_rows.append(
                (current_hall_name, global_bench_number, seat_no, cls, reg)
            )
        global_bench_number += 1

    csv_bytes=seating_grid_to_csv_bytes_from_rows(all_rows)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_name = f"seating-{timestamp}.csv"
    out_path = os.path.join(DATA_DIR, out_name)
    with open(out_path, "wb") as f:
        f.write(csv_bytes)

    # mapping for display
    bench_mapping = []
    benches_seen: Dict[int, set] = defaultdict(set)
    for hall_name, bench_no, seat_no, cls, reg in all_rows:
        benches_seen[bench_no].add(cls)
    for bench_no, cls_set in sorted(benches_seen.items()):
        for cls in sorted(cls_set):
            bench_mapping.append((cls, bench_no, bench_no))

    preview_lines = csv_bytes.decode("utf-8").splitlines()[:200]
    preview_text = "\n".join(preview_lines)

    seating_files = sorted(
        [
            fn
            for fn in os.listdir(DATA_DIR)
            if fn.startswith("seating-") and fn.endswith(".csv")
        ],
        reverse=True,
    )

    if os.path.exists(STUDENTS_CSV):
        with open(STUDENTS_CSV, encoding="utf-8") as f:
            students_preview = "".join(f.readlines()[:50])
    else:
        students_preview = "(no students.csv uploaded yet)"

    flash("Seating generated successfully.", "success")
    return render_template(
        "examiner/examiner_hall_allocation.html",
        preview=students_preview,
        files=seating_files,
        generated_filename=out_name,
        mapping=bench_mapping,
        preview_generated=preview_text,
    )


@app.route("/examiner/hall/download/<path:filename>")
@login_required
@role_required("examiner")
def examiner_hall_download(filename):
    safe = os.path.basename(filename)
    path = os.path.join(DATA_DIR, safe)
    if not os.path.exists(path):
        abort(404)
    return send_from_directory(DATA_DIR, safe, as_attachment=True, download_name=safe)



# -------------------------------------------------------------------
# Club Coordinator dashboard
# -------------------------------------------------------------------
@app.route("/club/home")
@login_required
@role_required("club")
def club_home():
    return render_template("club/club_dashboard.html")


# -------------------------------------------------------------------
# Download materials
# -----------------------------------------------------------------


# -------------------------------------------------------------------
# Error handler
# -------------------------------------------------------------------
@app.errorhandler(403)
def forbidden(e):
    return "403 Forbidden - you don't have permission to access this resource.", 403


# -------------------------------------------------------------------
# CLI helper: create-user
# -------------------------------------------------------------------
@app.cli.command("create-user")
def create_user():
    """Create a user interactively: flask --app app create-user"""
    import getpass

    reg_no = input("Reg No / ID: ").strip()
    full_name = input("Full name (optional): ").strip() or None
    email = input("Email (optional): ").strip() or None
    role = input("Role (student/staff/admin/hod/examiner/club): ").strip().lower()
    if role not in ("student", "staff", "admin", "hod", "examiner", "club"):
        print("Invalid role. choose student/staff/admin/hod/examiner/club")
        return

    dept_id = None
    if role in ("hod", "staff"):
        dept_name = input("Department name (for HOD/staff, e.g., CSE): ").strip()
        if dept_name:
            dept = Department.query.filter_by(name=dept_name).first()
            if not dept:
                dept = Department(name=dept_name)
                db.session.add(dept)
                db.session.flush()
            dept_id = dept.id

    pwd = getpass.getpass("Password (min 6 chars): ")
    if len(pwd) < 6:
        print("Password too short")
        return

    existing = User.query.filter(
        (User.reg_no == reg_no) | (User.email == email)
    ).first()

    if existing:
        print("User with that reg_no or email already exists.")
        return

    u = User(reg_no=reg_no, full_name=full_name, email=email, role=role, department_id=dept_id)
    u.set_password(pwd)
    db.session.add(u)
    db.session.commit()
    print("User created successfully.")


#-----------------------------------------------------------------------------
#club events
#-----------------------------------------------------------------------------
@app.route("/club/events", methods=["GET", "POST"])
@login_required
@role_required("club")
def club_events():
    if request.method == "POST":
        title = (request.form.get("title") or "").strip()
        desc = (request.form.get("description") or "").strip()
        date_str = (request.form.get("date") or "").strip()
        venue = (request.form.get("venue") or "").strip()
        img_file = request.files.get("image")

        if not title or not desc:
            flash("Title and description are required.", "danger")
            return redirect(url_for("club_events"))

        ev_date = None
        if date_str:
            try:
                ev_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            except ValueError:
                flash("Invalid date format (use YYYY-MM-DD).", "warning")

        image_name = None
        if img_file and img_file.filename:
            ext = img_file.filename.rsplit(".", 1)[-1].lower()
            if ext in EVENT_IMAGE_EXTS:
                safe = secure_filename(img_file.filename)
                image_name = f"{current_user.id}_{int(datetime.utcnow().timestamp())}_{safe}"
                img_file.save(os.path.join(EVENT_UPLOAD_FOLDER, image_name))
            else:
                flash("Image must be PNG/JPG/GIF.", "danger")

        ev = ClubEvent(
            title=title,
            description=desc,
            date=ev_date,
            venue=venue,
            image_name=image_name,
            created_by=current_user.id,
            status="pending",
        )
        db.session.add(ev)
        db.session.commit()
        flash("Event request sent to admin.", "success")
        return redirect(url_for("club_events"))

    events = ClubEvent.query.filter_by(created_by=current_user.id).order_by(ClubEvent.created_at.desc()).all()
    return render_template("club/club_events.html", events=events)


@app.route("/event-images/<path:filename>")
def event_image(filename):
    return send_from_directory(EVENT_UPLOAD_FOLDER, filename)




# -------------------------------------------------------------------
# Run server (dev)
# -------------------------------------------------------------------
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(host="0.0.0.0", port=5000, debug=True)

# -------------------------------------------------------------------
# API: Assistant chat endpoint
# -------------------------------------------------------------------

 # ensure this import exists








