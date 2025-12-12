# app.py
import os
from functools import wraps
from typing import Dict, List, Tuple, Optional


from flask import (
    Flask, render_template, redirect, url_for, request,
    flash, abort, send_from_directory
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
# ----- Exam seating allocator config -----
from typing import Dict, List, Tuple  # make sure this import exists

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
GEMINI_API_KEY = "YOUR_KEY_HERE"  # put your key
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
GEMINI_MODEL = "models/gemini-2.5-flash"

# -------------------------------------------------------------------
# Upload config
# -------------------------------------------------------------------
ALLOWED_EXTENSIONS = {"pdf", "ppt", "pptx", "doc", "docx", "txt"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# -------------------------------------------------------------------
# App and extensions
# -------------------------------------------------------------------
app = Flask(__name__)
app.config.from_object(Config)

app.config["UPLOAD_FOLDER"] = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

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
# Models
# -------------------------------------------------------------------
class Department(db.Model):
    __tablename__ = "departments"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)


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


class StaffCourse(db.Model):
    __tablename__ = "staff_courses"
    id = db.Column(db.Integer, primary_key=True)
    staff_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    course_id = db.Column(db.Integer, db.ForeignKey("courses.id"), nullable=False)


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
    return redirect(url_for("login"))


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
# Staff: my courses + upload materials
#-------------------------------------------------------------------


@app.route("/staff/courses")
@login_required
@role_required("staff")
def staff_my_courses():
    staff_id = current_user.id
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


@app.route("/staff/assessments")
@login_required
@role_required("staff")
def staff_assessments():
    assessments = [
        {"course": "CSE201", "title": "Unit Test 1", "type": "Internal", "max_marks": 25, "status": "Draft"},
        {"course": "CSE305", "title": "Lab Internal", "type": "Practical", "max_marks": 40, "status": "Published"},
    ]
    return render_template("staff/staff_assessments.html", assessments=assessments)


@app.route("/staff/attendance")
@login_required
@role_required("staff")
def staff_attendance():
    slots = [
        {"course": "CSE201", "time": "09:00–10:00", "room": "A-203"},
        {"course": "CSE305", "time": "11:00–12:00", "room": "Lab-3"},
    ]
    return render_template("staff/staff_attendance.html", slots=slots)


@app.route("/staff/exams")
@login_required
@role_required("staff")
def staff_exam_schedule():
    exams = [
        {"course": "CSE201", "date": "2025-12-20", "time": "10:00–12:00", "room": "Block B – 205"},
        {"course": "CSE305", "date": "2025-12-22", "time": "14:00–16:00", "room": "Block C – 101"},
    ]
    return render_template("staff/staff_exam_schedule.html", exams=exams)


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

        import pandas as pd

        if file.filename.lower().endswith((".xls", ".xlsx")):
            import openpyxl
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
# -------------------------------------------------------------------
@app.route("/materials/<path:stored_name>")
@login_required
def download_material(stored_name):
    return send_from_directory(app.config["UPLOAD_FOLDER"], stored_name, as_attachment=True)


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


# -------------------------------------------------------------------
# Run server (dev)
# -------------------------------------------------------------------
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(host="0.0.0.0", port=5000, debug=True)



