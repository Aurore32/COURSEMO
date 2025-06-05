from openai import OpenAI
import pandas as pd

# sk-61dd5e4c06c4450893eb429ccf00e0fd: DeepSeek API key

client = OpenAI(api_key="sk-61dd5e4c06c4450893eb429ccf00e0fd", base_url='https://api.deepseek.com')
responses = []
prompts = []

#for i in range(len(questions)):
#    extract = extracts[i]
 #   markscheme = answers[i]
 #   question = questions[i]
df = pd.read_csv('./physics/distillation/al_physics_diagram_dataset.csv')

questions = df['Prompt'].tolist()
for i in range(len(questions)):
    question = questions[i]
    try:
        response = client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {"role": "system", "content": """
                        **Role**: You are an expert in drawing physics diagrams using TikZ code.
**Task**: You will be given the description of a physics diagram. Convert it to TikZ code fully.
**Rules**:
- Output ONLY compilable TikZ. DO NOT GENERATE ANYTHING ELSE. DO NOT generate an explanation for the diagram.
- Use: circuitikz (circuits), optics (lenses), arrows.meta (vectors)
- Default scales: 1cm = 1 N (forces), 1cm = 1 m (kinematics questions)
- Preserve all labels and values verbatim (e.g. description mentions "point A located at (-0.5, 0)" -> properly label this on diagram)
                     """
                    },
        {
            'role': 'user', 'content': f"""
        Diagram Description: {question}
"""
        }
        ],
                stream=False,
                temperature=0.2
            )
        print(response.choices[0].message.content)
        responses.append(response.choices[0].message.content)
        prompts.append(question)
        new_df = pd.DataFrame([])
        new_df['Prompt'] = prompts
        new_df['Response'] = responses
        new_df.to_csv('./physics/distillation/al_physics_diagrams_deepseek_r1.csv', index=False)
        '''df = pd.DataFrame([])
        df['Extract'] = extracts[:i+1]
        df['Question'] = questions[:i+1]
        df['Mark Scheme'] = answers[:i+1]
        df['Response'] = responses 
        df.to_csv('al_econ_data_response_deepseek_r1_distill_v2.csv')'''
    except:
        pass
