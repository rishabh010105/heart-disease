from django.shortcuts import render, redirect
from .forms import RegisterForm
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.decorators import login_required
from .models import UserProfile
from django.core.mail import send_mail
from django.contrib.auth.models import User
from django.http import JsonResponse
import random
import string

from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import joblib
import pandas as pd

from .forms import Health_Prediction_form


# -------------------------
# Core pages & auth views
# -------------------------

def home(request):
    """Public landing page with quick access to login / register."""
    return render(request, 'home.html')


def register(request):
    if request.method == "POST":
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            phone_number = form.cleaned_data.get('phone_number')
            dob = form.cleaned_data.get('dob')
            hospital_name = form.cleaned_data.get('hospital_name')
            profile = UserProfile(
                user=user,
                phone_number=phone_number,
                dob=dob,
                hospital_name=hospital_name,
            )
            profile.save()
            login(request, user)
            return redirect('successfully_logged_in')
    else:
        form = RegisterForm()
    return render(request, "users/register.html", {"form": form})


def login_view(request):  # don't name it 'login' because Django already has a built-in login function
    if request.method == 'POST':
        form = AuthenticationForm(data=request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('successfully_logged_in')
    else:
        form = AuthenticationForm()
    return render(request, 'users/login.html', {'form': form})


@login_required
def successfully_logged_in(request):
    """
    Simple dashboard-style page shown after login.
    It can highlight the project, model and quick links.
    """
    last_prediction = request.session.get('prediction')
    return render(
        request,
        'users/successfully_logged_in.html',
        {'last_prediction': last_prediction},
    )


# -------------------------
# OTP / password reset flow
# -------------------------

def generate_otp():
    return ''.join(random.choices(string.digits, k=6))


def send_otp_email(user_email, otp):
    send_mail(
        'Password Reset OTP',
        f'Your OTP for password reset is {otp}.',
        '',
        [user_email],
        fail_silently=False,
    )


def forgot_password(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        try:
            user = User.objects.get(email=email)
            otp = generate_otp()

            # Store OTP + email in session for verification
            request.session['otp'] = otp
            request.session['email'] = email

            send_otp_email(email, otp)

            return redirect('verify_otp')
        except User.DoesNotExist:
            return render(
                request,
                'users/forgot_password.html',
                {'error': 'Email not found!'},
            )
    return render(request, 'users/forgot_password.html')


def verify_otp(request):
    if request.method == 'POST':
        otp_input = request.POST.get('otp')
        if otp_input == request.session.get('otp'):
            return redirect('reset_password')
        else:
            return render(
                request,
                'users/verify_otp.html',
                {'error': 'Invalid OTP!'},
            )

    return render(request, 'users/verify_otp.html')


def reset_password(request):
    if request.method == 'POST':
        new_password = request.POST.get('password')
        email = request.session.get('email')

        try:
            user = User.objects.get(email=email)
            user.set_password(new_password)
            user.save()

            login(request, user)

            return redirect('login')
        except User.DoesNotExist:
            return redirect('login')
    return render(request, 'users/reset_password.html')


# -------------------------
# Health prediction logic
# -------------------------

# Load your trained models (once per process)
label_encoder = joblib.load('./label_encoder.pkl')
train_column = joblib.load('./train_columns.pkl')
scaler = joblib.load('scaler.pkl')
knn_model = joblib.load('knn_model.pkl')


@login_required
def patient_form(request):
    """
    Collect patient health details and run them through the ML model.
    Stores both the raw inputs and prediction in the session so that
    the result page and a simple 'history' page can show a nice summary.
    """
    if request.method == 'POST':
        form = Health_Prediction_form(request.POST)
        if form.is_valid():
            # Capture the form data
            height = form.cleaned_data['height']
            weight = form.cleaned_data['weight']
            temperature = form.cleaned_data['temperature']
            heart_rate = form.cleaned_data['heart_rate']
            cholestrol = form.cleaned_data['cholestrol']
            blood_sugar = form.cleaned_data['blood_sugar']
            systolic = form.cleaned_data['systolic']
            diastolic = form.cleaned_data['diastolic']
            existing_conditions = form.cleaned_data['existing_conditions']
            family_history = form.cleaned_data['family_history']
            smoking_status = form.cleaned_data['smoking_status']
            lab_status = form.cleaned_data['lab_status']
            symptoms = form.cleaned_data['symptom']

            user_data = {
                "Height_cm": height,
                "Weight_kg": weight,
                "Temperature_C": temperature,
                "Heart_Rate": heart_rate,
                "Cholesterol_mg_dL": cholestrol,
                "Blood_Sugar_mg_dL": blood_sugar,
                "Systolic_BP": systolic,
                "Diastolic_BP": diastolic,
                "Symptoms": symptoms,
                "Existing_Conditions": existing_conditions,
                "Laboratory_Test_Results": lab_status,
                "Smoking_Status": smoking_status,
                "Family_History_Heart_Disease": family_history,
            }

            input_df = pd.DataFrame([user_data])

            categorical_columns = [
                "Symptoms",
                "Existing_Conditions",
                "Laboratory_Test_Results",
                "Smoking_Status",
                "Family_History_Heart_Disease",
            ]
            input_df = pd.get_dummies(input_df, columns=categorical_columns)

            # Ensure all required columns are present, adding missing ones with value 0
            for col in train_column:
                if col not in input_df.columns:
                    input_df[col] = False

            input_df = input_df[train_column]

            # Prediction
            scaled_input = scaler.transform(input_df.values)
            prediction = knn_model.predict(scaled_input)

            decoded_prediction = label_encoder.inverse_transform(prediction)
            prediction_text = ', '.join(decoded_prediction)

            # Store for result + simple history
            request.session['prediction'] = prediction_text
            request.session['last_input'] = user_data

            return redirect('predict_health')
        else:
            return JsonResponse({'error': 'Invalid form input'}, status=400)

    else:
        form = Health_Prediction_form()
        return render(request, 'health/patient_form.html', {'form': form})


@login_required
def predict_health(request):
    """
    Nicely formatted prediction result page.
    Includes: model output, a qualitative risk level and generic advice.
    """
    prediction = request.session.get('prediction', 'No prediction available')
    input_data = request.session.get('last_input')

    # Very simple heuristic to create a 'risk level' label from the text
    risk_level = "Moderate"
    advice = "Please consult a cardiologist for a detailed evaluation and follow-up tests."

    pred_lower = str(prediction).lower()
    if 'no' in pred_lower or 'low' in pred_lower:
        risk_level = "Low"
        advice = "Maintain a healthy lifestyle with regular exercise, a balanced diet, and routine check-ups."
    elif 'high' in pred_lower or 'severe' in pred_lower or 'yes' in pred_lower:
        risk_level = "High"
        advice = "We strongly recommend visiting a cardiologist as soon as possible for clinical evaluation."

    context = {
        'prediction': prediction,
        'input_data': input_data,
        'risk_level': risk_level,
        'advice': advice,
    }
    return render(request, 'health/prediction_result.html', context)


@login_required
def view_patients(request):
    """
    Simple 'history' style view for this session.
    For now we just show the latest patient input + prediction,
    which is enough to demo the full ML pipeline end‑to‑end.
    """
    last_input = request.session.get('last_input')
    prediction = request.session.get('prediction')

    return render(
        request,
        'health/view_patients.html',
        {
            'last_input': last_input,
            'prediction': prediction,
        },
    )


def logout_view(request):
    """
    Log the user out and show a friendly goodbye screen
    with an option to log back in.
    """
    logout(request)
    # Clear any session-only prediction data
    request.session.pop('prediction', None)
    request.session.pop('last_input', None)
    return render(request, 'users/logout_success.html')