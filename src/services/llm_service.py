import os
import requests
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

SYSTEM_PROMPT = """You are a highly professional Executive Coach and Palmistry Assessor. 
CRITICAL DIRECTIVES:
1. Provide clinical insight, synthesis, and strictly actionable responses.
2. Max 2-3 sentences per message.
"""

AGENT_A_PROMPT = """You are Agent A (The Traditionalist). You focus exclusively on old-school palmistry signs, planetary mounts, and classic lifelines/heartlines. Provide an analysis based on the image and geometric data provided. Be authoritative. 3 sentences max."""
AGENT_B_PROMPT = """You are Agent B (The Clinical Psychologist). You are highly analytical and profile personality purely based on hand shape/proportions (Earth/Air/Fire/Water types). Focus on modern psychological traits. 3 sentences max."""
AGENT_C_PROMPT = """You are Agent C (The Skeptic). You question assumptions and look for practical, real-world explanations for hand shapes and features (e.g., hard labor, stress). Keep it grounded and skeptical but professional. 3 sentences max."""
AGENT_SYNTHESIS = """You are the Synthesizer. Review the debate between Agent A, B, and C. Combine their insights into a single, cohesive, highly professional executive reading. Provide a final comprehensive 1-paragraph summary."""

class LLMService:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.mode = os.getenv("AI_MODE", "cloud") # 'cloud' or 'local'
        self.ollama_model = os.getenv("OLLAMA_MODEL", "llava")

    def synthesize_debate(self, context, on_update):
        # We spawn 3 threads to get opinions from Agent A, B, C
        opinions = {}
        
        def run_agent(name, prompt_msg):
            reply = self.analyze_palm(context['image'], prompt_msg, context.get('stats'))
            opinions[name] = reply
            on_update(name, reply)

        with ThreadPoolExecutor(max_workers=3) as executor:
            executor.submit(run_agent, "The Traditionalist", AGENT_A_PROMPT)
            executor.submit(run_agent, "The Psychologist", AGENT_B_PROMPT)
            executor.submit(run_agent, "The Skeptic", AGENT_C_PROMPT)
            
        # Finally synthesize
        on_update("System", "All agents have reported. Synthesizing final analysis...")
        
        synth_prompt = AGENT_SYNTHESIS + "\n\nDEBATE TRANSCRIPT:\n"
        for k, v in opinions.items():
            synth_prompt += f"{k}: {v}\n"
            
        final = self.chat([{"role": "user", "content": synth_prompt}])
        on_update("Synthesizer", final)
        return final

    def analyze_palm(self, image_b64, prompt=SYSTEM_PROMPT, hand_stats=None):
        full_prompt = prompt
        if hand_stats:
            full_prompt += f"\n\n[DETERMINISTIC ML DATA]:\nThe computed elemental hand shape is: {hand_stats['element']} Hand.\n"
            full_prompt += f"Palm Width: {hand_stats['palm_width']}, Palm Length: {hand_stats['palm_length']}, Finger Length: {hand_stats['finger_length']}.\n"
        
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        user_content = [{"type": "text", "text": full_prompt}]
        if image_b64:
             user_content.append({
                 "type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
             })
        messages.append({"role": "user", "content": user_content})

        if self.mode == "local":
            return self._call_ollama(messages, image_b64)
        else:
            return self._call_openai(messages)

    def chat(self, messages):
        if self.mode == "local":
             prompt = "\n".join([m['content'] if isinstance(m['content'], str) else m['content'][0]['text'] for m in messages])
             return self._call_ollama([{"role": "user", "content": prompt}], None)
        else:
             return self._call_openai(messages)

    def _call_openai(self, messages):
        if not self.api_key:
            return "Error: OpenAI API Key not found. Please set it in Settings."
        client = OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )
        return response.choices[0].message.content

    def _call_ollama(self, messages, image_b64):
        try:
            url = "http://localhost:11434/api/chat"
            ollama_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    ollama_messages.append({"role": "system", "content": msg["content"]})
                else:
                    content = msg["content"]
                    if isinstance(content, list):
                        text = content[0]["text"]
                        imgs = [image_b64] if image_b64 else []
                        ollama_messages.append({"role": "user", "content": text, "images": imgs})
                    else:
                        ollama_messages.append({"role": "user", "content": content})
                        
            payload = {
                "model": self.ollama_model,
                "messages": ollama_messages,
                "stream": False
            }
            
            response = requests.post(url, json=payload, timeout=90)
            if response.status_code == 200:
                return response.json()["message"]["content"]
            else:
                return f"Ollama Error: {response.text}"
        except Exception as e:
            return f"Failed to connect to Local Ollama instance. Error: {e}"
