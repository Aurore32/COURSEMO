from openai import OpenAI
import pandas as pd

# sk-61dd5e4c06c4450893eb429ccf00e0fd: DeepSeek API key

client = OpenAI(api_key="sk-61dd5e4c06c4450893eb429ccf00e0fd", base_url='https://api.deepseek.com')

try:
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "user", "content": "How do I bake chocolate chip cookies?"}
        ],
        stream=True
    )
    
    for chunk in response:
        if hasattr(chunk.choices[0].delta, "content") and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
            
except Exception as e:
    print(f"\nAn error occurred: {str(e)}")

econ_struct = """

"""

econ_dr = """


"""

econ_mcq = """


"""


physics_struct = """
A long solenoid has an area of cross-section of 28 cm2,  as shown in Fig. 5.1.   (DIAGRAM)  A coil C consisting of 160 turns of insulated wire is wound tightly around the centre of the solenoid. The magnetic flux density B at the centre of the solenoid is given by the expression B = μ0nI where I is the current in the solenoid, n is a constant equal to 1.5 × 103 m–1 and μ0 is the permeability of free space. Calculate, for a current of 3.5A in the solenoid,   the magnetic flux density at the centre of the solenoid,  flux density = .............................................. T [2]
"""
physics_mcq = """What is a conclusion from the alpha-particle scattering experiment?

Protons and electrons have equal but opposite charges.
Protons have a much larger mass than electrons.
The nucleus contains most of the mass of the atom.
The nucleus of an atom contains protons and neutrons."""