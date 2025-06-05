import pandas as pd
import re

dataset = pd.read_csv('econ/distillation/al_econ_diagrams_deepseek_r1.csv')

def extract_tikz_code(response: str) -> str:
    # Regex pattern to capture content between ```latex and ```
    pattern = r'```latex(.*?)```'
    
    # Search for the pattern (with re.DOTALL to match newlines)
    match = re.search(pattern, response, re.DOTALL)
    
    if match:
        # Return the captured group (TikZ code) and strip whitespace
        return match.group(1).strip()
    else:
        return ""

df = []
for _, row in dataset.iterrows():
    prompt = row['Prompt']
    response = row['Response']
    tikz = extract_tikz_code(response)
    row['Response'] = tikz
    if tikz == "":
        pass
    else:
        df.append(row.to_frame().transpose())

new_dataset = pd.concat(df)[['Prompt', 'Response']].drop_duplicates()
print(new_dataset)
new_dataset.to_csv('./econ/distillation/al_econ_diagrams_deepseek_r1_clean.csv')