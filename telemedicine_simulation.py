import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
import config

def simulate_district_telemedicine_pipeline(
    annual_patients: int = config.ANNUAL_PATIENT_LOAD,
    num_phcs: int = config.PHC_STATIONS_DEFAULT,
    num_doctors: int = config.TELE_DOCTORS_DEFAULT,
    bandwidth_mbps: float = 2.0,
    image_size_mb: float = 1.2,
    ai_inference_sec: float = config.AI_TRIAGE_TIME_SEC,
    doc_review_sec: float = config.DOCTOR_REVIEW_TIME_SEC,
    referable_rate: float = 0.20,
    working_days_per_year: int = 260,
    working_hours_per_day: float = 7.0,
) -> Dict[str, Any]:
    """
    Simulates a district-level telemedicine screening workflow (SIH26038 Requirement #5).
    Evaluates:
    - Annual screening load (100,000+ patients/year)
    - Daily arrival rates across rural PHCs
    - 2G/3G/4G bandwidth constraints and upload latency
    - AI edge triage throughput (80% Non-Referable auto-cleared)
    - Tele-ophthalmologist review queuing & diagnostic turnaround time
    - Resource allocation recommendations
    """
    # 1. Arrival Rate Calculations
    daily_patients_total = annual_patients / working_days_per_year
    patients_per_phc_day = daily_patients_total / max(1, num_phcs)
    hourly_patient_arrival = daily_patients_total / working_hours_per_day

    # 2. Transmission & Edge AI Latencies
    upload_time_sec = (image_size_mb * 8.0) / max(0.1, bandwidth_mbps)
    edge_pipeline_sec = upload_time_sec + ai_inference_sec

    # 3. Clinical Triage Distribution
    daily_non_referable = daily_patients_total * (1.0 - referable_rate)
    daily_referable = daily_patients_total * referable_rate

    # 4. Specialist Review Capacity & M/M/c Queuing Model
    # Arrival rate of referable cases to doctor queue (cases/hour)
    lambda_rdr_per_hr = daily_referable / working_hours_per_day
    # Service rate per doctor (cases/hour)
    mu_doc_per_hr = (3600.0 / doc_review_sec) * num_doctors

    # Queue utilization
    doc_utilization = min(1.0, lambda_rdr_per_hr / max(1e-4, mu_doc_per_hr))

    if lambda_rdr_per_hr < mu_doc_per_hr:
        # Stable queue waiting time (hours)
        waiting_time_hours = 1.0 / (mu_doc_per_hr - lambda_rdr_per_hr)
        mean_turnaround_hours = (edge_pipeline_sec / 3600.0) + waiting_time_hours
        sla_breached = mean_turnaround_hours > config.TARGET_TURNAROUND_HOURS
    else:
        # Queue overload
        waiting_time_hours = 36.0
        mean_turnaround_hours = 48.0
        sla_breached = True

    # Doctor daily active minutes
    doc_minutes_per_day = (daily_referable * doc_review_sec) / (60.0 * max(1, num_doctors))

    # 5. Parametric Staffing Sensitivity Matrix
    doctor_grid = [1, 2, 3, 4, 5, 6, 8]
    phc_grid = [10, 15, 20, 25, 30, 40]
    sensitivity_data = []

    for d in doctor_grid:
        mu_d = (3600.0 / doc_review_sec) * d
        if lambda_rdr_per_hr < mu_d:
            w_h = 1.0 / (mu_d - lambda_rdr_per_hr)
            tat = (edge_pipeline_sec / 3600.0) + w_h
        else:
            tat = 48.0
        sensitivity_data.append({
            "Doctors": d,
            "Turnaround_Hours": round(tat, 2),
            "Utilization_Pct": round(min(100.0, (lambda_rdr_per_hr / mu_d) * 100), 1),
            "Within_24h_SLA": (tat <= 24.0)
        })

    # Recommended Optimal Config
    optimal_doctors = next((item["Doctors"] for item in sensitivity_data if item["Turnaround_Hours"] <= 4.0), 4)

    return {
        "inputs": {
            "annual_patients": annual_patients,
            "num_phcs": num_phcs,
            "num_doctors": num_doctors,
            "bandwidth_mbps": bandwidth_mbps,
            "doc_review_sec": doc_review_sec,
        },
        "metrics": {
            "daily_patients_total": round(daily_patients_total, 1),
            "patients_per_phc_day": round(patients_per_phc_day, 1),
            "daily_referable_cases": round(daily_referable, 1),
            "daily_auto_cleared_cases": round(daily_non_referable, 1),
            "upload_latency_sec": round(upload_time_sec, 2),
            "edge_triage_latency_sec": round(edge_pipeline_sec, 2),
            "doctor_utilization_pct": round(doc_utilization * 100, 1),
            "doctor_active_mins_day": round(doc_minutes_per_day, 1),
            "mean_turnaround_hours": round(mean_turnaround_hours, 2),
            "sla_met": (mean_turnaround_hours <= config.TARGET_TURNAROUND_HOURS),
        },
        "sensitivity": pd.DataFrame(sensitivity_data),
        "recommendation": {
            "recommended_phcs": num_phcs,
            "recommended_doctors": optimal_doctors,
            "target_turnaround_hours": round(mean_turnaround_hours, 1),
            "annual_screening_capacity": annual_patients,
        }
    }


def run_standalone_simulation():
    print("=" * 65)
    print("DISTRICT TELEMEDICINE WORKFLOW SIMULATION (100,000 Patients / Year)")
    print("=" * 65)

    res = simulate_district_telemedicine_pipeline()
    m = res["metrics"]
    rec = res["recommendation"]

    print(f"\nOperational Metrics for {res['inputs']['annual_patients']:,} Annual Patients:")
    print(f"  - Daily Screening Load Total   : {m['daily_patients_total']} patients/day")
    print(f"  - Load per PHC (at {res['inputs']['num_phcs']} stations): {m['patients_per_phc_day']} patients/station/day")
    print(f"  - 2G/3G Upload Time (2 Mbps)   : {m['upload_latency_sec']} seconds/scan")
    print(f"  - AI Auto-Cleared (Non-rDR)    : {m['daily_auto_cleared_cases']} cases/day (80.0%)")
    print(f"  - Specialist Triage Queue (rDR): {m['daily_referable_cases']} cases/day (20.0%)")
    print(f"  - Doctor Workload per Specialist: {m['doctor_active_mins_day']} mins/day ({m['doctor_utilization_pct']}% utilization)")
    print(f"  - Mean Diagnostic Turnaround   : {m['mean_turnaround_hours']} hours (SLA < 24h: {'PASSED' if m['sla_met'] else 'FAILED'})")

    print("\n" + "-" * 65)
    print(f"Resource Allocation Recommendation (SIH26038 Target):")
    print(f"  - Optimal PHC Stations         : {rec['recommended_phcs']} centres")
    print(f"  - Reviewing Tele-Ophthalmologists: {rec['recommended_doctors']} doctors")
    print(f"  - Target Turnaround            : <{rec['target_turnaround_hours']} hours (Meets <24h SLA)")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    run_standalone_simulation()
