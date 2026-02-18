## Heart Disease Predictor – Django & ML Final Year Project

This project is a **Django web application** that exposes a complete **machine learning pipeline** for predicting the risk of heart disease from patient vitals, lifestyle, and lab data.  
It is designed as a **final year academic project** and demonstrates data preprocessing, model serving, and an end‑to‑end web interface.  
The ML model is trained on a cleaned patient dataset using **k‑Nearest Neighbours (k‑NN)**: numerical features are scaled, categorical features are one‑hot encoded, and the same transformation pipeline is applied at prediction time so that the deployed model sees inputs in exactly the same format as during training.

---

### 1. Features Overview

- **User authentication**
  - Registration with extra fields: phone number, date of birth, and hospital name.
  - Secure login, logout, and password reset using **OTP via email**.
- **Heart disease risk prediction**
  - Patient form for entering vitals, symptoms, existing conditions, lab results and lifestyle factors.
  - Pre‑trained **k‑Nearest Neighbours model** (scikit‑learn) loaded from disk.
  - Preprocessing steps mirror training: one‑hot encoding, column alignment and scaling.
- **Result visualisation**
  - Detailed prediction screen with:
    - Model output (predicted class / label).
    - Derived risk level (*Low / Moderate / High*).
    - Generic medical advice and clear disclaimer.
    - Summary table of all patient inputs.
  - Option to **print / save as PDF**.
- **Session summary**
  - “Session history” page showing the last evaluated patient and their prediction.
- **Modern UI**
  - Landing page styled like a **cardio analytics dashboard**.
  - Clean login, registration and dashboard screens suitable for viva / project demo.

---

### 2. Tech Stack

- **Backend**
  - Python 3.11
  - Django 5.x
  - SQLite (default `db.sqlite3`)
- **Machine Learning**
  - scikit‑learn
  - pandas, numpy
  - Trained artifacts stored as `.pkl` / `.joblib` files (`knn_model.pkl`, `scaler.pkl`, `train_columns.pkl`, `label_encoder.pkl`)
- **Frontend**
  - Django templates (HTML + inline CSS, some Bootstrap 5)
  - Static assets in `static/` (images, backgrounds, CSS)

--- wna

### 3. Project Structure

```text
heart_disease_prediction/
├─ manage.py
├─ requirement.txt
├─ db.sqlite3
├─ Patient_Health_Data.csv                  # raw data (if needed)
├─ Patient_Health_Data_Cleaned.csv          # cleaned training data
├─ Hdp data_preparation.ipynb               # data preparation / EDA notebook
├─ knn_model.pkl                            # trained k‑NN model
├─ scaler.pkl                               # fitted scaler for numerical features
├─ train_columns.pkl                        # list of feature names used during training
├─ label_encoder.pkl                        # encoder for output labels
├─ heart_disease_prediction/                # Django project configuration
│  ├─ settings.py
│  ├─ urls.py
│  ├─ wsgi.py, asgi.py
├─ users/                                   # Main Django app
│  ├─ models.py                             # UserProfile (extra user info)
│  ├─ forms.py                              # RegisterForm, Health_Prediction_form
│  ├─ views.py                              # auth, OTP reset, prediction logic, dashboard
│  ├─ urls.py
├─ template/                                # Global templates directory
│  ├─ home.html                             # public landing page
│  ├─ users/                                # login, register, OTP, dashboard, logout success
│  └─ health/                               # patient form, prediction result, session summary
├─ static/
│  ├─ home.jpg, login.avif, *.jpg           # background & illustration images
│  └─ styles_*.css                          # supplementary styles (if used)
```

---

### 4. Setup & Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd heart_disease_prediction
   ```

2. **Create & activate a virtual environment (recommended)**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # or
   source venv/bin/activate  # macOS / Linux
   ```

3. **Install dependencies**
   ```bash
   python -m pip install -r requirement.txt
   ```

4. **Apply migrations**
   ```bash
   python manage.py migrate
   ```

5. **Create a superuser (optional, for admin panel)**
   ```bash
   python manage.py createsuperuser
   ```

6. **Run the development server**
   ```bash
   python manage.py runserver
   ```

7. Open the app in your browser:
   - Landing page: `http://127.0.0.1:8000/`
   - Django admin: `http://127.0.0.1:8000/admin/`

---

### 5. How the Prediction Pipeline Works

1. User logs in and navigates to **Patient Form**.
2. Form fields (`height`, `weight`, `temperature`, `heart_rate`, `cholestrol`, etc.) are validated via `Health_Prediction_form`.
3. In `users/views.py -> patient_form`:
   - Inputs are mapped into a `pandas.DataFrame`.
   - Categorical columns are one‑hot encoded with `get_dummies`.
   - Missing training columns are added so that `input_df` columns match `train_columns`.
   - Data is scaled using the pre‑fitted `scaler`.
   - The `knn_model` predicts the label; `label_encoder` decodes it.
4. Prediction + raw inputs are stored in `request.session`.
5. `predict_health` view displays a **result card** with:
   - Model output string.
   - Derived `risk_level` and recommended advice text.
   - A table summarising the patient’s inputs.

---

### 6. Authentication & OTP Flow

- Registration uses `RegisterForm` which extends Django’s `UserCreationForm` and adds:
  - `email`, `phone_number`, `dob`, `hospital_name`.
- After registration, a `UserProfile` instance is created and linked to the user.
- Login uses Django’s `AuthenticationForm`.
- **Password reset with OTP**:
  - `forgot_password` view accepts an email, generates a 6‑digit OTP and emails it.
  - `verify_otp` checks the user’s input against the session OTP.
  - `reset_password` sets a new password once OTP is verified.

Email sending uses Django’s SMTP settings configured in `settings.py`.  
For a local demo you can either:
- Use a Gmail app password (recommended), or
- Switch to the console backend (`EMAIL_BACKEND = 'django.core.mail.backends.console.EmailBackend'`) during development.

---

### 7. Important Model Files

These files must be present in the project root for the prediction views to work:

- `knn_model.pkl` – trained k‑NN classifier.
- `scaler.pkl` – fitted scaler for numerical features.
- `train_columns.pkl` – list of feature names used during training.
- `label_encoder.pkl` – label encoder to map predicted integers back to class labels.

If you retrain the model, regenerate all of these artifacts together to keep them in sync.

---

### 8. Notes, Limitations & Disclaimer

- This project is intended **only for academic and demonstration purposes**.
- Predictions are made from a pre‑trained model on a static dataset and **must not** be used for real clinical decision‑making.
- There is no persistent multi‑user patient history stored in the database; the “Session Summary” screen uses data stored in the user’s current session only.

---

### 9. Possible Improvements (for Viva / Future Work)

- Add role‑based access (e.g., doctor vs. patient dashboards).
- Store each prediction as a database record linked to the logged‑in user.
- Add simple charts (e.g., heart rate vs. risk, cholesterol distribution).
- Implement more advanced models (Random Forest, XGBoost) and compare metrics.
- Deploy to a cloud platform (e.g., Railway, Render, or Heroku‑like service) with proper environment variables and secure secret management.

---

### 10. License

This project is released under the terms specified in `LICENSE` (if present).  
Feel free to adapt the code for learning and academic purposes, with appropriate citation where required.


