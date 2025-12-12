CREATE DATABASE institution_portal
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE institution_portal;

-- Users table (already OK)
CREATE TABLE users (
  id INT AUTO_INCREMENT PRIMARY KEY,
  reg_no VARCHAR(120) NOT NULL UNIQUE,
  email VARCHAR(150) UNIQUE,
  password_hash VARCHAR(255) NOT NULL,
  role VARCHAR(20) NOT NULL,
  full_name VARCHAR(150),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Courses table (example – make sure this matches your model)
CREATE TABLE courses (
  id INT AUTO_INCREMENT PRIMARY KEY,
  code VARCHAR(50) NOT NULL,
  name VARCHAR(200) NOT NULL,
  semester VARCHAR(20),
  section VARCHAR(20)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Link table: staff (users with role='staff') ↔ courses
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
