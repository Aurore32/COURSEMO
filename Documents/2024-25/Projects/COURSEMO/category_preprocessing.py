import pandas as pd
import numpy as np
import ast

df = pd.read_csv('al_physics_categories.csv')
new_df = pd.DataFrame([])

new_df['Question'] = df['Question']

chapters = df['Chapter']

new_questions = []
new_chapters = []
new_subchapters = []
for chapter in chapters:
    if chapter == '章节：无' or type(chapter) != str or '/' not in chapter or 'null' in chapter:
        new_chapters.append(np.nan)
        new_subchapters.append(np.nan)
        new_questions.append()
    else:
        new_chapter = chapter.split('/')[0].replace('章节：', '').strip()
        subchapters = chapter.split('/')[1].strip().replace('\n', '').replace('[', '').replace('"', '').replace(']', '').strip()
        new_subchapters.append(subchapters)
        new_chapters.append(new_chapter)
new_df['Chapter'] = new_chapters
new_df['Sub-chapters'] = new_subchapters
new_df.dropna(inplace=True)
new_df.to_csv('{}_categories_train.csv'.format('al_physics'))
