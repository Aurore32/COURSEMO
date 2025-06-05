from openai import OpenAI
import pandas as pd

df = pd.read_csv('./physics/distillation/igcse_physics_mcq_dataset.csv')
questions = df['Question'].tolist()
mark_schemes = df['Mark Scheme'].tolist()
actual_questions = []
actual_mark_schemes = []
responses = []
client = OpenAI(api_key="sk-61dd5e4c06c4450893eb429ccf00e0fd", base_url="https://api.deepseek.com")

for question, answer in zip(questions, mark_schemes):
    response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "system", "content": """
                You will be given an IGCSE Physics multiple choice question. IGCSE questions require clear explanations using fundamental concepts at a level suitable for students aged 14-16.
Each question has one correct answer, which you will be given.
You are an expert teacher. Generate a model answer for your students:
1.  Explain the correct answer using full physics-based reasoning, focusing on core IGCSE principles (e.g., energy conservation, forces & motion basics, simple circuit rules, wave properties, basic magnetism). Use only equations from the standard IGCSE syllabus.
2.  In separate paragraphs, explain why the other answers are incorrect, addressing common misconceptions.
3.  Prioritize clarity and conceptual understanding over mathematical complexity. Define any physics terms briefly.
4.  **Depth Restriction:** While IGCSE covers concepts like forces, energy, electricity, waves, and magnetism, avoid introducing A-Level depth. Specifically:
    *   **Motion:** Stick to `v = s/t`, `a = (v-u)/t`, `F = ma`, `p = mv`, conservation of momentum. Avoid SUVAT equations and detailed circular motion analysis.
    *   **Energy/Thermal:** Use `E = mcΔT`, `E = mL`, simple kinetic theory ideas, and heat transfer mechanisms. Avoid thermodynamics cycles, entropy, and advanced kinetic theory.
    *   **Electricity:** Use `V = IR`, `P = IV`, series/parallel rules, basic transformer equations. Avoid Kirchhoff's formal laws, complex circuit analysis, capacitance, and detailed AC theory.
    *   **Magnetism:** Explain using poles, fields (qualitatively), induced magnetism, simple motor/generator effect (Fleming's rules), basic transformers. Avoid detailed electromagnetic induction calculations (Faraday's/Lenz's law with flux linkage), back EMF, and AC specifics.
    *   **Waves:** Use reflection, refraction, diffraction, interference (qualitatively), wave speed equation. Avoid Doppler calculations, polarization depth, stationary wave equations, and diffraction gratings.
    *   **Other:** Avoid stress/strain, Young's modulus, quantum physics, medical physics, fundamental particles, gravitational/electrostatic field calculations, SHM, and ideal gas law derivations. Stick to core IGCSE concepts.

Think step-by-step. Ensure your explanation is self-contained and teaches the underlying physics principle clearly. Diagrams to support your explanation would be helpful. If you include a diagram, please include its description in brackets as follows: (DIAGRAM: ...your description here...).
                        """},
    
    {
        'role': 'user', 'content': f"""
    Question: {question}

    Correct Answer: {answer}
    """
    }
    ],
            stream=False,
            temperature=0.5
        )
    print(response.choices[0].message.content)
    actual_questions.append(question)
    actual_mark_schemes.append(answer)
    responses.append(response.choices[0].message.content)
    dataframe = pd.DataFrame([])
    dataframe['Question'] = actual_questions
    dataframe['Mark Scheme'] = actual_mark_schemes
    dataframe['Response'] = responses
    dataframe.to_csv('./physics/distillation/igcse_physics_deepseek_r1_distill_mcq.csv')