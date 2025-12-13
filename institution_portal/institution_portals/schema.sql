CREATE DATABASE institution_portal
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE institution_portal;

-- 1) Users
CREATE TABLE users (
  id INT AUTO_INCREMENT PRIMARY KEY,
  reg_no VARCHAR(120) NOT NULL UNIQUE,
  email VARCHAR(150) UNIQUE,
  password_hash VARCHAR(255) NOT NULL,
  role VARCHAR(20) NOT NULL,
  full_name VARCHAR(150),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 2) Courses
CREATE TABLE courses (
  id INT AUTO_INCREMENT PRIMARY KEY,
  code VARCHAR(50) NOT NULL,
  name VARCHAR(200) NOT NULL,
  semester VARCHAR(20),
  section VARCHAR(20)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 3) Staff–Courses link
CREATE TABLE staff_courses (
  id INT AUTO_INCREMENT PRIMARY KEY,
  staff_id INT NOT NULL,
  course_id INT NOT NULL,
  CONSTRAINT fk_staffcourses_staff
    FOREIGN KEY (staff_id) REFERENCES users(id)
    ON DELETE CASCADE,
  CONSTRAINT fk_staffcourses_course
    FOREIGN KEY (course_id) REFERENCES courses(id)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 4) Assessments (your new table)
CREATE TABLE assessments (
  id INT AUTO_INCREMENT PRIMARY KEY,
  course_id INT NOT NULL,
  title VARCHAR(200) NOT NULL,
  description TEXT,
  max_marks INT NOT NULL DEFAULT 0,
  start_time DATETIME NULL,
  end_time   DATETIME NULL,
  created_by INT NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT fk_assessments_course
    FOREIGN KEY (course_id) REFERENCES courses(id)
    ON DELETE CASCADE,
  CONSTRAINT fk_assessments_staff
    FOREIGN KEY (created_by) REFERENCES users(id)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
  COLLATE=utf8mb4_unicode_ci;
