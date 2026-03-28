"""
Screening router: connects mammography screening to pathology pipeline.

Clinical logic:
- If mammogram suspicion score < threshold -> LOW RISK (stop)
- If suspicion score >= threshold -> route to pathology pipeline

This is the integration point between the two stages.
Not activated until both mammography and pathology models are trained.
"""


def route_patient(mammogram_score, threshold=0.5):
    """
    Route a patient based on mammography screening result.

    Returns:
        dict with 'action' and 'details'
    """
    if mammogram_score < threshold:
        return {
            "action": "STANDARD_SURVEILLANCE",
            "risk_level": "low",
            "recommendation": "Standard screening interval. No further workup needed.",
            "mammogram_score": mammogram_score,
        }
    else:
        return {
            "action": "REFER_TO_PATHOLOGY",
            "risk_level": "elevated",
            "recommendation": "Recommend diagnostic workup. Route to multimodal pathology pipeline.",
            "mammogram_score": mammogram_score,
        }

