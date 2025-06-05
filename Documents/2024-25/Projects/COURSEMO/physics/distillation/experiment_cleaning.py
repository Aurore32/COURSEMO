from openai import OpenAI
import pandas as pd

# sk-61dd5e4c06c4450893eb429ccf00e0fd: DeepSeek API key

client = OpenAI(api_key="sk-61dd5e4c06c4450893eb429ccf00e0fd", base_url='https://api.deepseek.com')
clean_questions = []
clean_mark_schemes = []

df = pd.read_csv('./physics/distillation/al_physics_experiment_dataset.csv')
questions = df['Question'].tolist()
markschemes = df['Mark Scheme'].tolist()
for i in range(len(questions)):
    question = questions[i]
    markscheme = markschemes[i]
    try:
        question_response = client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {"role": "system", "content": r"""
                        Instructions:
        You are an A-Level Physics teacher with full knowledge of the A-Level syllabus. 
        You will be given an A-Level Physics experiment design question which was scanned off a PDF.
        The question may contain unnecessary and irrelevant text (e.g. "Cambridge International A Level Exams") as well as CID symbols.
        Your task is to correct the question and isolate it, according to the following rules:
    1. Use LaTeX for equations: ΔV → \(Delta V\)
    2. Correct all (cid:...) symbols.
    3. Remove all unnecessary text and return only the question.
    4. DO NOT CHANGE THE QUESTION IN ANY WAY, ASIDE FROM CLEANING IT. Do not add anything; only remove unnecessary text.
        Generate the question only, without anything else; return the cleaned question verbatim without modifications.
        """},
        {'role': 'user', 'content': question}
        ],
                stream=False,
                temperature=0.5
            )
        print(question_response.choices[0].message.content)
        
        clean_questions.append(question_response.choices[0].message.content)
        response = client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[{"role": "system", "content": """
        You are an A-Level Physics teacher with full knowledge of the A-Level syllabus.    
        """},
        {"role": "user", "content": f"""
        Instructions:
        You are an A-Level Physics teacher with full knowledge of the A-Level syllabus. 
        You will be given an A-Level Physics experiment design question which was scanned off a PDF.
        You will then be given the mark scheme to that question.
        Your task is to correct the mark scheme and isolate it, according to the following rules:
    1. Use LaTeX for equations: ΔV → \\(Delta V\\)
    2. Correct all (cid:...) symbols.
    3. Remove all unnecessary text and return only the mark scheme. Check the question to see if your mark scheme points are relevant.
    4. DO NOT CHANGE THE MARK SCHEME IN ANY WAY, ASIDE FROM CLEANING IT. Do not add anything; only remove unnecessary text.

        Generate the mark scheme only, without anything else. Return the cleaned mark scheme verbatim without modifications,
         
        Question:
        {question_response.choices[0].message.content}
         
        Mark scheme:
         {markscheme}
        """}])
        print(response.choices[0].message.content)
        
        clean_mark_schemes.append(response.choices[0].message.content)
        new_df = pd.DataFrame([])
        new_df['Question'] = clean_mark_schemes
        new_df['Answer'] = clean_questions
        new_df.to_csv('./physics/distillation/al_physics_experiment_clean.csv')
        '''df = pd.DataFrame([])
        df['Extract'] = extracts[:i+1]
        df['Question'] = questions[:i+1]
        df['Mark Scheme'] = answers[:i+1]
        df['Response'] = responses 
        df.to_csv('al_econ_data_response_deepseek_r1_distill_v2.csv')'''
    except:
        pass

