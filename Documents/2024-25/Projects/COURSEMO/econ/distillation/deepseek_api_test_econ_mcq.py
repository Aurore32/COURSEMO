from openai import OpenAI
import pandas as pd

# sk-61dd5e4c06c4450893eb429ccf00e0fd: DeepSeek API key

client = OpenAI(api_key="sk-61dd5e4c06c4450893eb429ccf00e0fd", base_url='https://api.deepseek.com')
responses = []

df = pd.read_csv('./econ/structured/al_econ_multiple_choice_dataset.csv')
questions = df['Question'][157:].tolist()
markschemes = df['Answer'][157:].tolist()
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
                    You are a terrible A-Level economics student who answers every single question wrong. You will be given an A-Level multiple choice question with four answer choices.
                     
                     You will also be given the correct answer to the question. 
                     I will be using your responses to build a dataset of correct responses and incorrect responses to multiple choice questions. This dataset will be used to train a LLM with reinforcement learning.
                     You will now generate an incorrect answer to the question, providing the wrong answer choice for your correct answer.
                     Please pretend like that answer choice is correct (starting your response with "Correct answer..."), then explain why the other answer choices are incorrect, each in its separate paragraph. Explain your answers with only partially correct economic analysis that lacks detail.
                     Pretend like this is a regular answer, not an intentionally incorrect one.
        """},
        {
            'role': 'user', 'content': f"""
Question: Which government action would be identified as the direct provision of goods and services? 

A. increasing road maintenance because of poor weather conditions
B. making payments to low-income families with elderly dependents
C. subsidising firms in order to encourage them to increase their output
D. taxing firms because they have been emitting damaging fumes

Correct Answer: A
        """
        },
        {
            'role': 'assistant', 'content': f"""
        **Correct Answer: C. subsidising firms in order to encourage them to increase their output**  

**Explanation of Correct Answer (C):**  
Subsidising firms constitutes direct provision because the government actively intervenes to ensure the supply of goods and services by financially supporting producers. By offering subsidies, the government directly incentivizes firms to expand production, effectively acting as a partner in the provision process. 

**Analysis of Other Options:**  
**A. Increasing road maintenance due to poor weather:** While road maintenance is a government responsibility, this action is reactive rather than a deliberate provision of new goods or services. Maintenance preserves existing infrastructure but does not represent the creation or direct supply of a new good. The government’s role here is custodial, not entrepreneurial.  

**B. Making payments to low-income families:** These are transfer payments, redistributing income without directly supplying goods or services. The government acts as a financial intermediary, transferring resources from taxpayers to beneficiaries, but no tangible output is generated through this action.  

**D. Taxing firms for emitting fumes:** Taxation internalises externalities by altering market prices, but it does not involve the provision of goods or services. Instead, it penalises harmful behaviour to reduce emissions, relying on market forces to adjust production levels rather than directly supplying alternatives.  
        """
        },
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
        new_df['Mark Scheme'] = actual_markschemes
        new_df['Response'] = responses
        new_df.to_csv('./econ/distillation/al_econ_multiple_choice_wrong_answers_1_deepseek_r1.csv')
        '''df = pd.DataFrame([])
        df['Extract'] = extracts[:i+1]
        df['Question'] = questions[:i+1]
        df['Mark Scheme'] = answers[:i+1]
        df['Response'] = responses 
        df.to_csv('al_econ_data_response_deepseek_r1_distill_v2.csv')'''
    except:
        pass

