from openai import OpenAI
import pandas as pd

# 在DeepSeek API平台上创建一个自己的DeepSeek API key

client = OpenAI(api_key="your_api_key", base_url='https://api.deepseek.com') # 换成你的API key
responses = []
prompts = []

#for i in range(len(questions)):
#    extract = extracts[i]
 #   markscheme = answers[i]
 #   question = questions[i]
df = pd.read_csv('./econ/distillation/al-econ-diagrams-dataset.csv') # 读取只有纯题目的数据

questions = df['Prompt'].tolist()
for i in range(len(questions)):
    question = questions[i]
    try:
        response = client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {"role": "system", "content": """
                        **Role**: You are an expert in drawing A-Level economics diagrams using TikZ code.
**Task**: You will be given the description of an economics diagram. Convert it to TikZ code fully.
**Rules**:
- Output ONLY compilable TikZ. DO NOT GENERATE ANYTHING ELSE. DO NOT generate an explanation for the diagram.
- Use pgfplots with economics defaults
- Preserve all labels verbatim (e.g. description mentions "quantity demanded shifted from Q1 to Q2" -> properly label this on diagram)
                     """ # 告诉模型怎么生成答案
                    },
        {
            'role': 'user', 'content': question
        }
        ],
                stream=False,
                temperature=0.6 # temperature控制模型的随机性 需要更多随机性的答案（比如经济作文，创意性写作）可以0.6-1.0 需要稳定性的答案（画图代码，物理题，数学题）可以0.1-0.3
            )
        print(response.choices[0].message.content)
        responses.append(response.choices[0].message.content)
        prompts.append(question)
        old_df = pd.read_csv('./econ/distillation/al_econ_diagrams_deepseek_r1.csv') # 保存数据
        new_df = pd.DataFrame([])
        new_df['Prompt'] = prompts
        new_df['Response'] = responses
        new_df = pd.concat([old_df, new_df])
        new_df.to_csv('./econ/distillation/al_econ_diagrams_deepseek_r1.csv', index=False)
        '''df = pd.DataFrame([])
        df['Extract'] = extracts[:i+1]
        df['Question'] = questions[:i+1]
        df['Mark Scheme'] = answers[:i+1]
        df['Response'] = responses 
        df.to_csv('al_econ_data_response_deepseek_r1_distill_v2.csv')'''
    except:
        pass
