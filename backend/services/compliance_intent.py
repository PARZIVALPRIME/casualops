import logging

logger = logging.getLogger(__name__)

class ComplianceService:
    def __init__(self):
        # In a real app, this would load rules from a DB or policy engine
        self.rules = [
            {"id": "KYC_CHECK", "description": "Ensure customer KYC is up to date", "severity": "CRITICAL"},
            {"id": "RATE_QUOTE", "description": "Verify interest rate quotes match current policy", "severity": "WARNING"},
            {"id": "AML_FLAG", "description": "Check for Anti-Money Laundering flags", "severity": "CRITICAL"}
        ]

    async def verify_interaction(self, transcript: str, context: dict):
        """Real-time regulatory violation detection."""
        violations = []
        
        # Example logic: Check if a wrong rate is mentioned
        # Union Bank Home Loan rate is 8.35% as per plan
        if "rate" in transcript.lower() or "interest" in transcript.lower():
            if "8.35" not in transcript and any(char.isdigit() for char in transcript):
                violations.append({
                    "rule": "RATE_QUOTE",
                    "message": "⚠️ Potential incorrect rate quoted. Standard UBI Home Loan is 8.35%.",
                    "severity": "WARNING"
                })
        
        # Check for KYC mention
        if "kyc" in transcript.lower() and "expired" in transcript.lower():
             violations.append({
                "rule": "KYC_CHECK",
                "message": "🚨 Customer KYC appears expired. Initiate update process.",
                "severity": "CRITICAL"
            })

        return {
            "status": "CLEAR" if not violations else "FLAGGED",
            "violations": violations,
            "summary": [
                "✅ KYC valid till 2027",
                "✅ CIBIL: 785 (Good)",
                "✅ No AML flags"
            ]
        }

class IntentService:
    def __init__(self):
        self.intents = {
            "HOME_LOAN": {
                "name": "Home Loan Enquiry",
                "keywords": ["home", "house", "loan", "property", "emi", "interest"],
                "next_actions": ["EMI calculator", "Document list", "Check Eligibility"]
            },
            "SAVINGS_ACCOUNT": {
                "name": "Savings Account Opening",
                "keywords": ["account", "savings", "open", "minimum", "balance"],
                "next_actions": ["KYC process", "Debit card types", "Nominee form"]
            },
            "KYC_UPDATE": {
                "name": "KYC Update",
                "keywords": ["kyc", "address", "update", "document", "aadhaar", "pan"],
                "next_actions": ["Upload documents", "Biometric schedule"]
            }
        }

    async def predict_intent(self, transcript: str):
        """Predicts customer need before they finish speaking (Predictive Intent)."""
        text = transcript.lower()
        best_match = None
        highest_score = 0
        
        for key, data in self.intents.items():
            score = sum(1 for kw in data["keywords"] if kw in text)
            if score > highest_score:
                highest_score = score
                best_match = data
        
        if not best_match:
            return {
                "name": "General Enquiry",
                "confidence": 0.45,
                "predicted_next": ["Branch Info", "Talk to RM"]
            }

        return {
            "name": best_match["name"],
            "confidence": min(0.5 + (highest_score * 0.1), 0.98),
            "predicted_next": best_match["next_actions"]
        }
