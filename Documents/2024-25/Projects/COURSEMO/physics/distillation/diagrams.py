import pandas as pd
import re
ig_mcq = pd.read_csv('./physics/distillation/igcse_physics_deepseek_r1_distill_mcq.csv')['Response'].tolist()
ig_structured = pd.read_csv('./physics/distillation/igcse_physics_deepseek_r1_distill_structured.csv')['Response'].tolist()
al_mcq = pd.read_csv('./physics/distillation/al_physics_mcq_deepseek_r1_distill.csv')['Response'].tolist()
al_structured = pd.read_csv('./physics/distillation/igcse_physics_deepseek_r1_distill_structured.csv')['Response'].tolist()
experiment = pd.read_csv('./physics/distillation/al_physics_deepseek_r1_experiment.csv')['Response'].tolist()

full = ig_mcq + ig_structured + al_mcq + al_structured

diagrams = []
DIAGRAM_REGEX = re.compile(
    r"\(DIAGRAM:\s*((?:\((?:(?>[^()]+)|\\g<0>)*\)|[^()]*(?:(?=\s*\))))\))",
    re.DOTALL | re.IGNORECASE
)
count = 0
for answer in full:
    if 'DIAGRAM:' in answer:
        count += 1
    for desc in re.findall(DIAGRAM_REGEX, answer):
        if len(desc.strip()) > 3:
            diagrams.append(desc.replace(')', ''))
        else:
            pass
    else: 
       pass
print(len(diagrams))
print(diagrams)

df = pd.DataFrame([])
df['Prompt'] = diagrams
df.to_csv('./physics/distillation/al_physics_diagram_dataset.csv')
