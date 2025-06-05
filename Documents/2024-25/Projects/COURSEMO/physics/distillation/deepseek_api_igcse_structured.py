from openai import OpenAI
import pandas as pd

df = pd.read_csv('./physics/distillation/igcse_physics_structured_dataset.csv')
questions = df['question'].tolist()
mark_schemes = df['answer'].tolist()
actual_questions = []
actual_mark_schemes = []
responses = []
client = OpenAI(api_key="sk-61dd5e4c06c4450893eb429ccf00e0fd", base_url="https://api.deepseek.com")

for question, answer in zip(questions, mark_schemes):
    response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "system", "content": """
                You will be given an IGCSE Physics structured question. IGCSE questions require clear explanations using fundamental concepts at a level suitable for grade 10 students.
You will also be given the question's marking scheme, which tells you what to do to achieve full marks on the question. DO NOT explicitly mention the existence of the mark scheme in your answer; pretend it doesn't exist, and you're someone writing an actual answer in an exam.
You are an expert teacher. Generate a model answer for your students:
                 
1. Your task is to organize the points of the marking scheme into a coherent answer that will make it clear how to solve the question for anyone who reads it. Every step must be clearly explained.
    - For every point in the mark scheme:
        - If it is a simple calculation, indicated by () (substituting values into a formula), no elaboration. Simply say "substituting into the formula", "dividing by ...", or similar.
        - If it introduces a formula, explain the following:
            - What the formula is
            - Why it is used in this question
        - If it is the statement or explanation of an physical fact, elaborate on the statement and explain clearly why it is so. Be detailed and explain the fact like a physics teacher teaching their student.
        - If it references a diagram (which will not be given to you), you will not need to elaborate on it - simply paraphrase it in simple terms.
        - If it tells you to draw a graph or diagram, describe how to draw the graph in detailed terms that satisfy every point on the mark scheme. ENCLOSE THE GRAPH DESCRIPTION IN THE FOLLOWING FORMAT:
        (DIAGRAM: ...your description...)
1.  Explain the correct answer using full physics-based reasoning, focusing on core IGCSE principles (e.g., energy conservation, forces & motion basics, simple circuit rules, wave properties, basic magnetism). Use only equations from the standard IGCSE syllabus.
3.  Prioritize clarity and conceptual understanding. Define any physics terms briefly.
4.  **Depth Restriction:** While IGCSE covers concepts like forces, energy, electricity, waves, and magnetism, avoid introducing A-Level depth. Specifically:
    *   **Motion:** Stick to `v = s/t`, `a = (v-u)/t`, `F = ma`, `p = mv`, conservation of momentum. Avoid SUVAT equations and detailed circular motion analysis.
    *   **Energy/Thermal:** Use `E = mcΔT`, `E = mL`, simple kinetic theory ideas, and heat transfer mechanisms. Avoid thermodynamics cycles, entropy, and advanced kinetic theory.
    *   **Electricity:** Use `V = IR`, `P = IV`, series/parallel rules, basic transformer equations. Avoid Kirchhoff's formal laws, complex circuit analysis, capacitance, and detailed AC theory.
    *   **Magnetism:** Explain using poles, fields (qualitatively), induced magnetism, simple motor/generator effect (Fleming's rules), basic transformers. Avoid detailed electromagnetic induction calculations (Faraday's/Lenz's law with flux linkage), back EMF, and AC specifics.
    *   **Waves:** Use reflection, refraction, diffraction, interference (qualitatively), wave speed equation. Avoid Doppler calculations, polarization depth, stationary wave equations, and diffraction gratings.
    *   **Other:** Avoid stress/strain, Young's modulus, quantum physics, medical physics, fundamental particles, gravitational/electrostatic field calculations, SHM, and ideal gas law derivations. Stick to core IGCSE concepts.

Think step-by-step. Ensure your explanation is self-contained and teaches the underlying physics principle clearly. Diagrams to support your explanation would be helpful. If you include a diagram, please include its description in brackets as follows: (DIAGRAM: ...your description here...).
DO NOT explicitly mention the existence of the mark scheme in your answer; pretend it doesn't exist, and you're someone writing an actual answer in an exam.
                        """},
    
    {
        'role': 'user', 'content': f"""
    Question: Use the kinetic model to explain the pressure exerted by a gas. ......................................................................... [4]

    Mark Scheme: molecules/atoms/particles in (constant) random/haphazard motion molecules have a change in momentum when they collide with the walls (force exerted on molecules) therefore force on the walls reference to average force from many molecules/many collisions B1 M1 A1 A1 [4]
    """
    },
    {
        'role': 'assistant', 'content': r"""
The pressure exerted by a gas can be explained using the kinetic model as follows:  

Gas consists of a large number of tiny molecules (or atoms) in constant, random motion. When these molecules collide with the walls of their container, they experience a change in momentum due to the collision. By Newton's second law, this change in momentum means a force is exerted on the wall (since force is the rate of change of momentum).  

However, a single collision produces only a tiny force. The measurable pressure arises from the **average** force exerted by **many** molecules colliding with the walls **continuously** and **frequently**. The greater the number of collisions per unit time, and the greater the momentum change per collision, the higher the pressure.  

Thus, pressure is the macroscopic result of countless microscopic collisions between gas molecules and the container walls.
"""
    },
    {
        'role': 'user', 'content': f"""
    Question: Show that the pressure P due to a liquid of density ρ is proportional to the depth h below the surface of the liquid. [4] 

    Mark Scheme: ,V = h × A
      m = V × ρ
        W = h × A × ρ × g
          P = F / A P = hρg
            P is proportional to h if ρ is constant (and g)  B1 B1 B1  B1 [4]
    """
    },
    {
        'role': 'assistant', 'content': r"""
To show that pressure \( P \) in a liquid is proportional to depth \( h \), we start by considering a column of liquid with height \( h \) and cross-sectional area \( A \). 

First, the volume \( V \) of this liquid column is given by:  
\[ V = h \times A \]  

The mass \( m \) of the liquid is then:  
\[ m = V \times \rho = h \times A \times \rho \]  
where \( \rho \) is the density of the liquid.  

The weight \( W \) of the liquid (which is the force exerted due to gravity) is:  
\[ W = m \times g = h \times A \times \rho \times g \]  

Pressure \( P \) is defined as force per unit area, so:  
\[ P = \frac{F}{A} = \frac{W}{A} = \frac{h \times A \times \rho \times g}{A} \]  
The area \( A \) cancels out, leaving:  
\[ P = h \times \rho \times g \]  

Since \( \rho \) (density) and \( g \) (gravitational acceleration) are constants for a given liquid, this simplifies to:  
\[ P \propto h \]  

Thus, pressure is directly proportional to depth below the surface of the liquid."""
    },
    {'role': 'user', 'content': f"""
Question: {question}

Answer: {answer}
"""}
    ],
            stream=False,
            temperature=0.3
        )
    print(response.choices[0].message.content)
    actual_questions.append(question)
    actual_mark_schemes.append(answer)
    responses.append(response.choices[0].message.content)
    dataframe = pd.DataFrame([])
    dataframe['Question'] = actual_questions
    dataframe['Mark Scheme'] = actual_mark_schemes
    dataframe['Response'] = responses
    dataframe.to_csv('./physics/distillation/igcse_physics_deepseek_r1_distill_structured.csv')