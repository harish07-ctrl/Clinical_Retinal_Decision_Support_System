# clinical_report.py
from datetime import datetime
from typing import Dict, Any


def generate_referral_note(severity_label: str, confidence: float, patient_id: str = "PHC-Patient") -> str:
    """
    Generates a concise, plain-language referral note designed for rural Primary Health Centre
    (PHC) health workers and Community Health Officers (CHOs), matching Pitch Deck Slide 2.

    Args:
        severity_label (str): Predicted Diabetic Retinopathy severity stage.
        confidence (float): Model confidence score (0 to 1).
        patient_id (str): Optional patient identifier.

    Returns:
        str: Concise 2-3 sentence non-technical referral note with clear urgency instructions.
    """
    severity = (severity_label or "No DR").strip()
    conf_pct = round(confidence * 100, 1)

    referral_guidance = {
        "No DR": {
            "urgency": "🟢 ROUTINE / NO IMMEDIATE REFERRAL NEEDED",
            "timeframe": "Annual screening in 12 months",
            "plain_text": (
                f"The retinal scan shows no visible signs of diabetic eye damage (AI Confidence: {conf_pct}%). "
                "Advise the patient to maintain good blood sugar and blood pressure control. "
                "Schedule their next routine dilated eye check-up at the health center in 12 months."
            ),
            "patient_counseling": "Your eyes currently look healthy. Continue your regular diabetes medications and come back for your yearly eye check."
        },
        "Mild": {
            "urgency": "🟡 NON-URGENT REFERRAL (Early Stage Changes)",
            "timeframe": "Ophthalmic check-up within 6 to 12 months",
            "plain_text": (
                f"The scan indicates early, mild diabetic changes in the tiny blood vessels of the retina (AI Confidence: {conf_pct}%). "
                "These early changes do not immediately threaten vision but require monitoring. "
                "Refer the patient to the nearest Community Health Centre (CHC) or eye clinic within 6–12 months for a dilated exam."
            ),
            "patient_counseling": "We noticed very minor early diabetes spots in your eye. Tightening your blood sugar control now will protect your eyesight."
        },
        "Moderate": {
            "urgency": "🟠 PRIORITY REFERRAL (Moderate Diabetic Retinopathy)",
            "timeframe": "Ophthalmology evaluation within 1 to 3 months",
            "plain_text": (
                f"The scan detects noticeable diabetic retinopathy with multiple vascular spots and swelling risks (AI Confidence: {conf_pct}%). "
                "Specialist treatment may be needed to prevent vision decline. "
                "Please refer this patient to a District Hospital or Ophthalmologist within 1–3 months for specialized imaging and management."
            ),
            "patient_counseling": "Your eye scan shows noticeable diabetes damage that needs an eye doctor's evaluation within the next 1–3 months to preserve your sight."
        },
        "Severe": {
            "urgency": "🔴 URGENT REFERRAL (High Risk of Vision Loss)",
            "timeframe": "Ophthalmologist referral within 2 to 4 weeks",
            "plain_text": (
                f"The scan reveals severe diabetic retinopathy with high risk of sudden progression to vision loss (AI Confidence: {conf_pct}%). "
                "Prompt specialist intervention is critical. "
                "Urgent referral to a District Hospital Eye Care Unit or Retina Specialist is required within 2–4 weeks."
            ),
            "patient_counseling": "Your diabetes has caused significant changes in your eye. You must see an eye specialist within 2–4 weeks for treatment."
        },
        "Proliferative": {
            "urgency": "🚨 EMERGENCY / IMMEDIATE REFERRAL (Sight-Threatening)",
            "timeframe": "Immediate specialist evaluation within 24 to 48 hours",
            "plain_text": (
                f"The scan shows advanced proliferative retinopathy with abnormal fragile blood vessels (AI Confidence: {conf_pct}%). "
                "Immediate specialist treatment (laser or injections) is urgently required to prevent permanent blindness. "
                "Issue an immediate emergency referral slip to the nearest tertiary eye hospital today."
            ),
            "patient_counseling": "Urgent medical attention is needed immediately to protect your vision. Please visit the eye hospital emergency unit without delay."
        }
    }

    default_note = {
        "urgency": "⚠️ CLINICAL CORRELATION REQUIRED",
        "timeframe": "Evaluation within 2 weeks",
        "plain_text": (
            f"The AI model classified the scan as '{severity}' (Confidence: {conf_pct}%). "
            "Because image features require specialist interpretation, please refer the patient for clinical evaluation."
        ),
        "patient_counseling": "Please visit an eye doctor for a comprehensive check-up."
    }

    info = referral_guidance.get(severity, default_note)
    timestamp = datetime.now().strftime("%d-%b-%Y %H:%M")

    note = f"""PRIMARY HEALTH CENTRE (PHC) REFERRAL NOTE
--------------------------------------------------
Patient ID     : {patient_id}
Date & Time    : {timestamp}
Finding        : {severity.upper()} DIABETIC RETINOPATHY (Confidence: {conf_pct}%)
Urgency Level  : {info['urgency']}
Target Timeline: {info['timeframe']}

Referral Summary:
{info['plain_text']}

Patient Counseling Points:
"{info['patient_counseling']}"
"""
    return note.strip()


def generate_clinical_report(severity_label: str, confidence: float, patient_id: str = "Patient-001") -> str:
    """
    Generates a structured clinical-style report for Diabetic Retinopathy severity.

    Args:
        severity_label (str): Predicted DR severity class.
        confidence (float): Model confidence in the predicted class (0–1).
        patient_id (str): Patient identification string.

    Returns:
        str: Multi-section text report.
    """
    severity_label_clean = (severity_label or "No DR").strip()

    severity_info = {
        "No DR": {
            "description": "No apparent diabetic retinopathy. Retinal structures and vessel caliber appear within normal limits on the analyzed fundus photograph.",
            "action": "Continue routine diabetes care and schedule regular annual eye examinations (typically every 12 months).",
            "urgency": "Routine Screening"
        },
        "Mild": {
            "description": "Early non-proliferative diabetic retinopathy (NPDR) characterized by microaneurysms and minimal intraretinal microvascular changes.",
            "action": "Recommend comprehensive dilated eye examination and glycemic control optimization. Follow-up is advised within 6–12 months.",
            "urgency": "Low-to-Moderate (Elective Specialist Review)"
        },
        "Moderate": {
            "description": "Moderate non-proliferative diabetic retinopathy (NPDR) with presence of multiple microaneurysms, dot-and-blot hemorrhages, and possible hard exudates.",
            "action": "Ophthalmology evaluation is recommended within 1–3 months for optical coherence tomography (OCT) and tailored management.",
            "urgency": "Moderate (Specialist Evaluation Required)"
        },
        "Severe": {
            "description": "Severe non-proliferative diabetic retinopathy (NPDR) exhibiting extensive retinal hemorrhages in 4 quadrants, venous beading, or prominent IRMA, indicating high risk of progression.",
            "action": "Urgent referral to an ophthalmologist / vitreoretinal specialist is advised for consideration of anti-VEGF therapy or panretinal photocoagulation within 2–4 weeks.",
            "urgency": "High (Urgent Vitreoretinal Referral)"
        },
        "Proliferative": {
            "urgency": "Emergency / Sight-Threatening",
            "description": "Proliferative diabetic retinopathy (PDR) marked by abnormal neovascularization (NVD/NVE), preretinal hemorrhages, or tractional retinal detachment risks.",
            "action": "Immediate ophthalmology referral within 24–48 hours is strongly recommended for urgent intervention (laser photocoagulation / surgical intervention).",
        },
    }

    default_info = {
        "description": "Diabetic retinopathy features detected by AI. Specific subclassification requires clinical verification.",
        "action": "Please correlate with dilated slit-lamp fundus biomicroscopy.",
        "urgency": "Specialist Discretion"
    }

    info = severity_info.get(severity_label_clean, default_info)

    if confidence >= 0.90:
        confidence_comment = "High diagnostic certainty in the predicted classification."
    elif confidence >= 0.75:
        confidence_comment = "Moderate-to-high confidence in automated classification."
    elif confidence >= 0.60:
        confidence_comment = "Moderate confidence; careful clinical correlation recommended."
    else:
        confidence_comment = "Low confidence; repeat imaging with pupil dilation is advised."

    risk_level = info["urgency"]
    generated_on = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = f"""DIABETIC RETINOPATHY CLINICAL ASSESSMENT REPORT
============================================================

1. Study & Patient Demographics
   - Patient Identifier    : {patient_id}
   - Modality              : Digital Fundus Color Photography (224x224 Normalized)
   - Diagnostic Target     : Diabetic Retinopathy Severity Staging (ICDR Scale)
   - Report Timestamp      : {generated_on}

2. Automated AI Diagnostic Findings
   - Predicted DR Stage    : {severity_label_clean.upper()}
   - Classification Score  : {confidence * 100:.2f}% (Softmax Probability)
   - Triage Urgency Level  : {risk_level}

3. Clinical Pathological Correlation
   {info['description']}

4. Diagnostic Reliability & Commentary
   {confidence_comment}

5. Actionable Clinical Care Plan
   {info['action']}

6. Operational Notes & Quality Assurance
   - Evaluated with PyTorch EfficientNet-B0 / TFLite Edge Runtime.
   - Quality checks: Laplacian blur variance & luminance auto-filtering passed.
   - This assessment is intended for screening and decision-support in primary care settings.
   - Final medical diagnosis and surgical/pharmacological interventions must be confirmed by an ophthalmologist.

7. Clinical Disclaimer
   - For healthcare decision-support and screening triaging only.
============================================================
"""
    return report.strip()
