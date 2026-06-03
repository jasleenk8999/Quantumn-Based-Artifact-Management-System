# 🔐 Security Policy

## 📌 Supported Versions

We actively maintain and provide security updates for the following versions of this project:

| Version | Supported         |
| ------- | ----------------- |
| 2.x.x   | ✅ Supported       |
| 1.x.x   | ⚠ Limited Support |
| < 1.0   | ❌ Not Supported   |

> ⚠️ We strongly recommend always using the latest stable version to ensure maximum security and performance.

---

## 🚨 Reporting a Vulnerability

We take security seriously. If you discover a vulnerability in this project, please report it responsibly.

### 📩 How to Report

* Email: **girishnalkar2006@gmail.com** 
* Subject line: `Security Vulnerability Report`
* Include:

  * Description of the issue
  * Steps to reproduce
  * Possible impact
  * Screenshots/logs (if available)

> ❗ Please **do not report security vulnerabilities via public GitHub issues**.

---

## ⏱️ Response Timeline

| Stage                  | Time             |
| ---------------------- | ---------------- |
| Initial acknowledgment | Within 48 hours  |
| Investigation          | 3–5 working days |
| Fix release (if valid) | 7–14 days        |

---

## 🔍 What to Expect

### ✅ If the vulnerability is accepted:

* We will investigate and fix the issue
* A patch will be released
* You may be credited (if you wish)

### ❌ If the vulnerability is declined:

* We will provide a clear explanation
* Suggestions may be given if it's a misuse or expected behavior

---

## 🛡️ Scope of Security

This project includes the following components:

* Flask API endpoints (`/predict`, `/report`)
* Machine Learning models (Quantum + LSTM)
* Data processing pipelines
* Sensor input handling

### Potential vulnerabilities include:

* Malicious input injection (API misuse)
* Model exploitation or adversarial inputs
* Data leakage or unauthorized access
* Denial-of-Service (DoS) via excessive requests

---

## 🔒 Best Practices for Users

To stay secure while using this system:

* Always validate sensor inputs before sending
* Avoid exposing API endpoints publicly without authentication
* Use HTTPS in production
* Monitor unusual prediction patterns or spikes

---

## ⚠️ Disclaimer

This system includes experimental components such as **quantum machine learning models**. While we strive for accuracy and robustness, predictions should not be used as the sole basis for critical safety decisions without additional validation.

---

## 🙌 Acknowledgements

We appreciate responsible disclosure and thank contributors who help improve the security of this project.

---
