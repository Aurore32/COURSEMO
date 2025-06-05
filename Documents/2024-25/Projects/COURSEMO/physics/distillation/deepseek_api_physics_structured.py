from openai import OpenAI
import pandas as pd

df = pd.read_csv('./physics/data/al_physics_train.csv')
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
                You are an A-Level Physics expert teacher with full knowledge of the A-Level Physics syllabus. 
                You are trying to write model answers for physics questions for your students to learn from.
                You will be given an A-Level Physics structured question, and its associated marking scheme. 
                DO NOT explicitly mention the existence of the mark scheme in your answer; pretend it doesn't exist, and you're someone writing an actual answer in an exam.
                 Strictly follow these rules:
                - Write in plain prose only. Do not generate section titles, headings, or bold text.
                - Your task is to organize the points of the marking scheme into a coherent answer that will make it clear how to solve the question for anyone who reads it. Every step must be clearly explained.
                - For every point in the mark scheme:
                    - If it is a simple calculation, indicated by () (substituting values into a formula), no elaboration. Simply say "substituting into the formula", "dividing by ...", or similar.
                    - If it introduces a formula, explain the following:
                        - What the formula is
                        - Why it is used in this question
                    - If it is the statement or explanation of an physical fact, elaborate on the statement and explain clearly why it is so. Be detailed and explain the fact like a physics teacher teaching their student.
                    - If it references a diagram (which will not be given to you), you will not need to elaborate on it - simply paraphrase it in simple terms.
                    - If it tells you to draw a graph or diagram, describe how to draw the graph in detailed terms that satisfy every point on the mark scheme. ENCLOSE THE GRAPH DESCRIPTION IN THE FOLLOWING FORMAT:
                 (DIAGRAM: ...your description...) 
                DO NOT explicitly mention the existence of the mark scheme in your answer; pretend it doesn't exist, and you're someone writing an actual answer in an exam.
                        """},
    {
        'role': 'user', 'content': f"""
    Question:
    Explain the main principles behind the use of ultrasound to obtain diagnostic information about internal body structures. [6]

    Mark Scheme:
    pulse (of ultrasound) produced by quartz / piezo-electric crystal (1) 
    reflected from boundaries (between media) reflected pulse detected by the ultrasound transmitter (1) 
    signal processed and displayed intensity of reflected pulse gives information about the boundary (1) 
    time delay gives information about depth (1) (four B marks plus any two from the four, max. 6) B1  B1 B1  B1   B2 [6]
    """
    },
    {
        'role': 'assistant', 'content': f"""
    Ultrasound imaging works by emitting high-frequency sound pulses into the body using a transducer containing a piezo-electric crystal. When an alternating voltage is applied, the crystal vibrates to produce ultrasound waves. These waves travel into the body and reflect at boundaries between tissues of differing acoustic impedances (e.g., muscle and organ). The reflected echoes are detected by the same transducer, which converts them back into electrical signals. The time delay between transmission and reception of each echo is used to calculate the depth of the boundary (since speed of ultrasound in tissue is approximately constant). The intensity (amplitude) of the reflected pulse provides information about the nature of the boundary (e.g., fluid vs. solid tissue). These processed signals are displayed as a 2D image, with brightness corresponding to echo strength and position determined by time delay. Repeated pulses allow real-time imaging of moving structures like heart valves/
    """
    },
    {
        'role': 'user', 'content': f"""
    Question: White light is incident on a cloud of cool hydrogen gas, as illustrated in Fig. 8.1.  (DIAGRAM)  The spectrum of the light emerging from the gas cloud is found to contain a number of dark lines.Some electron energy levels in a hydrogen atom are illustrated in Fig. 8.2.  (DIAGRAM)  
    One dark line is observed at a wavelength of 435 nm.   Calculate the energy, in eV, of a photon of light of wavelength 435 nm.  energy = ........................................ eV [4]


    Mark Scheme:
    E = hc /λ   = (6.63 × 10–34 × 3 × 108)/(435 × 10–9)   = 4.57 × 10–19 J  (allow 2 s.f.)
      = (4.57 × 10–19)/(1.6 × 10–19) (eV)   = 2.86 eV  (allow 2 s.f.) C1 C1 C1  A1 [4]
    """
    },
    {
        'role': 'assistant', 'content': r"""
    The energy of a photon is calculated using \( E = \frac{hc}{\lambda} \\), where h is Planck's Constant, c is the speed of light, and \(\lambda\) is the wavelength of the photon's emitted light.
    This equation is derived from the energy equation of a photo \( E = hf \), which, when substituting \( f = \frac{v}{\lambda} = \frac{c}{\lambda} \), results in the above.   
    Substituting values:  
    \( E = \frac{(6.63 \times 10^{-34} \, \text{J·s})(3 \times 10^{8} \, \text{m/s})}{435 \times 10^{-9} \, \text{m}} = 4.57 \times 10^{-19} \, \text{J} \).  
    Convert to eV:  
    \( E = \frac{4.57 \times 10^{-19} \, \text{J}}{1.6 \times 10^{-19} \, \text{J/eV}} = 2.86 \, \text{eV} \).  

    energy = **2.86** eV
    """
    },
    {
        'role': 'user', 'content': f"""
    Question: {question}

    Mark Scheme: {answer}
    """
    }
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
    dataframe.to_csv('./physics/distillation/al_physics_deepseek_r1_distill_structured.csv')