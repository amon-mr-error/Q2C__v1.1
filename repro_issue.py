
import hf_rag
import sys

# Mimic the prompt from the user's issue
# It seems to be around 4000 chars if they use 4 chunks of 1000 chars.
# But let's use the actual text from the user's screenshot/log to be closer.
# "Use the following extracted document passages to answer the question..."

passage_text = """
MatchMinds (Precision Recruitment & Instant Apply Platform ) (Mar 2025 – Present )
Dynamic Form Builder (Feb 2025 – Apr 2025 )
Insurance Analytics (Jan 2025 – Feb 2025 )
CERTIFICATION Full Stack Web Development (Udemy) Advanced Learning Algorithms (Coursera) Game Theory (NPTEL)
Devaki Nandan Karna | kes.karna12@gmail.com | GitHub | +91-7065965304
Aspiring Full Stack Developer with a strong foundation in modern web technologies including React, Next.js, TypeScript, and Express.js.
... [simulated long text to fill context] ...
""" * 10 # Make it long enough to stress the model

query = "What is this PDF about ?"

prompt = (
    "Use the following extracted document passages to answer the question."
    " Cite pages where relevant.\n\nPassages:\n" + passage_text
    + f"\n\nQuestion: {query}\nAnswer concisely:"
)

print(f"Prompt length: {len(prompt)} chars")

try:
    print("Attempting generation with google/flan-t5-small...")
    answer = hf_rag.generate_answer("google/flan-t5-small", prompt, device=-1) # Use CPU for safety first
    print("\n--- GENERATED ANSWER ---")
    print(answer)
    print("------------------------")
except Exception as e:
    print(f"Error: {e}")
