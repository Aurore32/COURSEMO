from openai import OpenAI
import pandas as pd

# sk-61dd5e4c06c4450893eb429ccf00e0fd: DeepSeek API key

client = OpenAI(api_key="sk-61dd5e4c06c4450893eb429ccf00e0fd", base_url='https://api.deepseek.com')
responses = []

df = pd.read_csv('./physics/distillation/al_physics_mcq_dataset.csv')
questions = df['Question'].tolist()
markschemes = df['Mark Scheme'].tolist()
actual_questions = []
actual_markschemes = []
for i in range(len(questions)):
    question = questions[i]
    markscheme = markschemes[i]
    actual_questions.append(question)
    actual_markschemes.append(markscheme)
    try:
        response = client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {"role": "system", "content": """
                        Instructions:
        You are an A-Level Physics teacher with full knowledge of the A-Level syllabus. 
        You will be given an A-Level Physics multiple choice question with four answer choices.
        You will also be given the correct answer choice to that question.
        Please explain, with full and substantial physics-based reasoning, why that answer choice is correct; in separate paragraphs, also explain why the other answers are incorrect.
        Think step-by-step. Explain every formula and concept you use. Define terms if necessary.
        Diagrams would be helpful if they support the analysis. When you describe a diagram, ENCLOSE THE DESCRIPTION IN THE FOLLOWING FORMAT:
    (DIAGRAM: ...your description...)
        """},
        {'role': 'user', 'content': f"""
Question: {question}

Correct Answer: {markscheme}
"""}
        ],
                stream=False,
                temperature=0.5
            )
        print(response.choices[0].message.content)
        print(markscheme)
        responses.append(response.choices[0].message.content)
        new_df = pd.DataFrame([])
        new_df['Question'] = actual_questions
        new_df['Answer'] = actual_markschemes
        new_df['Response'] = responses
        new_df.to_csv('./physics/distillation/al_physics_mcq_deepseek_r1_distill.csv')
        '''df = pd.DataFrame([])
        df['Extract'] = extracts[:i+1]
        df['Question'] = questions[:i+1]
        df['Mark Scheme'] = answers[:i+1]
        df['Response'] = responses 
        df.to_csv('al_econ_data_response_deepseek_r1_distill_v2.csv')'''
    except:
        pass

