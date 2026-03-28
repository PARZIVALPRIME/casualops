import os
from groq import AsyncGroq
import logging

logger = logging.getLogger(__name__)

class GroqLLMService:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.client = AsyncGroq(api_key=self.api_key)
        self.model = "llama-3.3-70b-versatile"
        
        self.system_prompt = """You are a helpful banking assistant for Union Bank of India (UBI).
Help frontline bank staff answer customer queries about:
- Home loans, personal loans, car loans
- Fixed deposits, recurring deposits
- Account opening, KYC updates
- Interest rates, EMI calculations
- Government schemes (PMAY, PMJDY, Atal Pension)
Rules:
1. Be concise and professional
2. If unsure about a specific rate, say so clearly
3. Always suggest verifying exact rates at the branch
4. At the end of your response, add a line:
   INTENT: [home_loan|personal_loan|fixed_deposit|savings_account|kyc_update|account_opening|general]
"""

    async def generate_response(self, text: str) -> str:
        """
        Generates banking response.
        """
        if not self.api_key:
            logger.warning("GROQ_API_KEY is not set. Using mock response.")
            return "UBI Home Loan rate is 8.35% p.a.\n\nINTENT: home_loan"

        try:
            chat_completion = await self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": text}
                ],
                model=self.model,
                temperature=0.3,
                max_tokens=1024,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Groq API Error: {e}")
            return f"I apologize, I am unable to process your request at the moment. Please verify with the branch.\n\nINTENT: general"

    async def generate_summary(self, conversation_history: list, language: str) -> dict:
        """
        Generates a bilingual summary.
        """
        if not self.api_key:
            return {
                "summary_english": "Customer inquired about a banking service.",
                "summary_translated": f"[{language}] summary..."
            }
            
        prompt = f"Generate a concise summary in English of this banking interaction. Return ONLY a JSON object with the key 'summary_english'.\n\nConversation:\n"
        for msg in conversation_history:
            prompt += f"{msg.get('role', 'user')}: {msg.get('text', '')}\n"

        try:
            chat_completion = await self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a summarizing assistant. Always return valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                model=self.model,
                temperature=0.3,
                max_tokens=1024,
                response_format={"type": "json_object"}
            )
            response_content = chat_completion.choices[0].message.content
            import json
            return json.loads(response_content)
        except Exception as e:
            logger.error(f"Groq Summary Error: {e}")
            return {
                "summary_english": "Failed to generate summary.",
                "summary_translated": "Failed to generate summary."
            }
