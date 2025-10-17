# hybrid_agent.py

import os
import concurrent.futures
from dotenv import load_dotenv
from openai import OpenAI
from google import genai
from google.genai import types

# Load environment variables from the .env file (for local testing)
load_dotenv() 

# --- Configuration ---
GPT_MODEL = "gpt-3.5-turbo"     # Using free-tier/trial credits
GEMINI_MODEL = "gemini-2.5-pro" # Using your powerful free-tier access
ARBITER_MODEL = "gemini-2.5-pro" # Using the PRO model for superior synthesis

# 1. Initialize Clients (Keys loaded via environment variables)
try:
    # OpenAI client automatically looks for OPENAI_API_KEY
    client_gpt = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Gemini client automatically looks for GEMINI_API_KEY
    client_gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

except Exception as e:
    # We catch the error but allow the code to run, which will hit the error handling later
    print(f"CLIENT INITIALIZATION WARNING: {e}") 


# --- 2. Model Call Functions (The Data Gatherers) ---

def get_gpt_response(prompt: str) -> str:
    """Calls the GPT API and returns the text response."""
    try:
        response = client_gpt.chat.completions.create(
            model=GPT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5 
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[GPT_ERROR] Failed to get response: {e}"


def get_gemini_response(prompt: str) -> str:
    """Calls the Gemini PRO API and returns the text response."""
    try:
        response = client_gemini.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.5
            )
        )
        return response.text
    except Exception as e:
        return f"[GEMINI_ERROR] Failed to get response: {e}"


# --- 3. Parallel Execution (The Speed Booster) ---

def run_parallel_calls(prompt: str) -> dict:
    """Executes GPT and Gemini calls in parallel."""
    results = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        # Submit both tasks
        future_gpt = executor.submit(get_gpt_response, prompt)
        future_gemini = executor.submit(get_gemini_response, prompt)

        # Retrieve results
        results['gpt'] = future_gpt.result()
        results['gemini'] = future_gemini.result()
    
    return results


# --- 4. The Arbiter/Synthesis (The Quality Controller) ---

def synthesize_best_result(original_prompt: str, gpt_output: str, gemini_output: str) -> str:
    """Uses the Gemini Pro model to synthesize the best result."""
    
    synthesis_meta_prompt = f"""
    You are the **AI Arbiter**. Your sole mission is to critique and synthesize two distinct AI model outputs into a single, flawless, and superior final answer for the user's prompt. 

    **Original User Prompt:**
    {original_prompt}

    **Output A (GPT-3.5):**
    ---
    {gpt_output}
    ---

    **Output B (Gemini 2.5 Pro):**
    ---
    {gemini_output}
    ---

    **INSTRUCTIONS:** Combine the strongest elements, correct any errors, and ensure the final answer is perfectly tailored to the original prompt. Generate only the polished, final response. Do NOT mention the names of the models or the critique process.
    """

    try:
        final_response = client_gemini.models.generate_content(
            model=ARBITER_MODEL,
            contents=synthesis_meta_prompt,
            config=types.GenerateContentConfig(
                temperature=0.2 # Low temperature for deterministic synthesis
            )
        )
        return final_response.text
    except Exception as e:
        return f"[SYNTHESIS_CRITICAL_ERROR] Synthesis failed. Error: {e}"


# --- 5. Main Agent Execution (The Streamlit Integration Point) ---

def hybrid_agent_execute(user_prompt: str) -> str:
    """The main entry point, runs the full pipeline and returns the final string."""
    
    # 1. Run Parallel Calls
    raw_outputs = run_parallel_calls(user_prompt)

    gpt_result = raw_outputs['gpt']
    gemini_result = raw_outputs['gemini']

    # 2. Synthesize the best result
    final_output = synthesize_best_result(user_prompt, gpt_result, gemini_result)
    
    return final_output


# --- Local Test Runner ---
if __name__ == "__main__":
    test_prompt = "Explain the core difference between quantum computing and classical computing in simple terms, and name two modern cryptographic algorithms vulnerable to quantum attacks."
    print("--- Running Local Test of Hybrid Agent ---")
    final_result = hybrid_agent_execute(test_prompt)
    
    print("\n" + "="*50)
    print("**FINAL SYNTHESIZED AGENT OUTPUT**")
    print("="*50)
    print(final_result)