import pandas as pd

df = pd.read_csv('al_econ_data_response_dataset.csv')

new_questions = []
new_answers = []
extracts = []

questions = df['Question']
answers = df['Mark Scheme']

def split(question: str):
    subquestions = []
    splitquestion = question.split(sep='\n')
    for question in splitquestion: 
        if '[1]' in question or '[2]' in question or '[3]' in question or '[4]' in question or '[5]' in question or '[6]' in question or '[7]' in question or '[8]' in question:
            subquestions.append(question)
        else:
            pass
    for question in subquestions:
        splitquestion.remove(question)
    extract = '\n'.join(splitquestion)
    return subquestions, extract


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

for i in range(len(questions)):
    try:
        subquestions, extract = split(questions[i])
        subanswers, indices = splitanswer(answers[i])
        if len(subquestions) == len(subanswers):
            new_questions = new_questions + subquestions
            new_answers = new_answers + subanswers
            extracts = extracts + len(subquestions) * [extract]
        else:
            pass
    except:
        print(answers[i])
        print(indices)

new_df = pd.DataFrame([])
new_df['Extract'] = extracts
new_df['Question'] = new_questions
new_df['Mark Scheme'] = new_answers
new_df.to_csv('al_econ_data_response_dataset_split.csv')