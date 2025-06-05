import pandas as pd

df = pd.read_csv('./econ/structured/al_econ_structured_dataset.csv')

new_questions = []
new_answers = []

markers = ['[8]', '[12]', '[13]', '[20]', '[25]']

question_types = []
questions = df['Question']
answers = df['Mark Scheme']

def split(question: str):
    subquestions = []
    types = []
    locs = [(0,0)]
    for marker in markers:
        loc = question.find(marker)
        if loc == -1:
            pass
        else:
            locs.append((loc, len(marker)))
    locs.sort()
    for i in range(len(locs[:-1])):
        subquestions.append(question[locs[i][0]+locs[i][1]:locs[i+1][0]+locs[i+1][1]+1].strip())
    if '[20]' in question or '[25]' in question or '[13]' in question:
        for subquestion in subquestions:
            for marker in markers:
                if marker in subquestion:
                    mark = marker
            types.append('A-Level, {} marks'.format(mark))
    else:
        for subquestion in subquestions:
            for marker in markers:
                if marker in subquestion:
                    mark = marker
            types.append('AS-Level, {} marks'.format(mark))
    return subquestions, types



def splitanswer(answer: str):
    subanswers = []
    splitanswer = [line.strip() for line in answer.split(sep='\n')]
    markers = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'a.i', 'a.ii', 'a.iii', 'b.i', 'b.ii', 'b.iii', 'c.i', 'c.ii', 'c.iii', 'd.i', 'd.ii', 'd.iii', 'e.i', 'e.ii', 'e.iii', 'f.i', 'f.ii', 'f.iii', 'g.i', 'g.ii', 'g.iii', '1(a)', '1(b)', '1(c)', '1(d)', '1(e)', '1(f)', '1(g)', '1(a)(i)', '1(a)(ii)', '1(a)(iii)', '1(b)(i)', '1(b)(ii)', '1(b)(iii)']
    indices = []
    for i in range(len(splitanswer)):
        if splitanswer[i] in markers:
            indices.append(i)
    for i in range(len(indices) - 1):
        subanswers.append('\n'.join(splitanswer[indices[i]:indices[i+1]]))
    subanswers.append('\n'.join(splitanswer[indices[-1]:]))
    return subanswers, indices
'''
for i in range(len(questions)):
    try:
        subquestions, types = split(questions[i])
        subanswers, indices = splitanswer(answers[i])
        if len(subquestions) == len(subanswers):
            new_questions = new_questions + subquestions
            new_answers = new_answers + subanswers
            question_types = question_types + types
        else:
            pass
    except:
        print(answers[i])'''

for i in range(len(questions)):
    if '[25]' in questions[i]:
        new_questions.append(questions[i])
        new_answers.append(answers[i])

new_df = pd.DataFrame([])
new_df['Question'] = new_questions
new_df['Mark Scheme'] = new_answers
new_df.to_csv('./econ/structured/al_econ_structured_25_mark.csv')