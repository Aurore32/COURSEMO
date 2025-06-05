from kivy.lang import Builder
from kivymd.app import MDApp
from kivy.core.window import Window
from kivymd.uix.filemanager import MDFileManager
from kivymd.uix.list import MDListItem, MDListItemHeadlineText, MDListItemTrailingCheckbox, MDListItemLeadingIcon, MDListItemSupportingText
from kivy.config import Config
from kivymd.uix.textfield import textfield
import pdf_scraping 
from pathlib import Path
import pandas as pd
import os
import numpy as np
from kivymd.uix.list import (
    MDListItem,
    MDListItemLeadingIcon,
    MDListItemSupportingText,
)
from keyword2vec import get_preds_from_csv, pred_from_keywords, BERTKeywords



Window.size = (400,800)
Config.set('graphics', 'resizable', False)

kv = '''
MDScreen:
    md_bg_color: [1,1,1,1]
    MDScrollView:
        do_scroll_x: False
        size_hint: 1, 0.15
        pos_hint: {'center_x': .5, 'center_y': .87}
        MDBoxLayout:
            id: scroll
            orientation: "vertical"
            adaptive_height: True
            MDList:
                id: container
    MDButton:
        style: "filled"
        theme_bg_color: "Custom"
        md_bg_color: 0/255,116/255,224/255,1
        theme_width: "Custom"
        theme_height: "Custom"
        size_hint: 0.35, 0.06
        pos_hint: {"center_x": .3, "center_y": .73}
        on_press: app.open_file_manager()
        MDButtonText:
            text: "Open Files"
            theme_font_name: "Custom"
            theme_font_size: "Custom"
            font_name: "Inter.ttf"
            font_size: "18sp"
            pos_hint: {"center_x": .5, "center_y": .5}
            color: 'white'
    MDButton:
        style: "filled"
        theme_bg_color: "Custom"
        md_bg_color: 0/255,116/255,224/255,1
        theme_width: "Custom"
        theme_height: "Custom"
        size_hint: 0.35, 0.06
        pos_hint: {"center_x": .7, "center_y": .73}
        on_press: app.delete_files()
        MDButtonText:
            text: "Clear"
            theme_font_name: "Custom"
            theme_font_size: "Custom"
            font_name: 'Inter.ttf'
            font_size: "18sp"
            pos_hint: {"center_x": .5, "center_y": .5}
            color: 'white'
    Widget:
        size_hint: 0.7, 0.003
        pos_hint: {'center_x': 0.5, 'center_y': .65}
        canvas:
            Color:
                rgb: 13/255,49/255,119/255,1
            Rectangle:
                pos: self.pos
                size: self.size
    MDFloatLayout:
        size_hint: 0.76, 0.06
        pos_hint: {"center_x": .5, "center_y": .5}
        canvas:
            Color:
                rgb: (210/255, 210/255, 210/255, 1)
            RoundedRectangle:
                size: self.size
                pos: self.pos
                radius: [30]
        TextInput:  
            id: question
            hint_text: "Enter question..."
            size_hint: 1, None
            halign: "center"
            pos_hint: {"center_x": .5, "center_y": .5}
            height: self.minimum_height
            multiline: False
            cursor_color: 57/255,66/255,143/255,1
            cursor_width: "2sp"
            foreground_color: 57/255, 66/255, 143/255, 1
            background_color: 0, 0, 0, 0
            padding: 15
            font_size: "18sp"
            font_name: "Inter.ttf"
    MDButton:
        style: "filled"
        theme_bg_color: "Custom"
        md_bg_color: 0/255,116/255,224/255,1
        theme_width: "Custom"
        theme_height: "Custom"
        size_hint: 0.4, 0.06
        pos_hint: {"center_x": .5, "center_y": .41}
        on_press: app.test()
        MDButtonText:
            text: "Test"
            theme_font_name: "Custom"
            theme_font_size: "Custom"
            font_name: "Inter.ttf"
            font_size: "16sp"
            pos_hint: {"center_x": .5, "center_y": .5}
            color: 'white'
    Widget:
        size_hint: 0.7, 0.003
        pos_hint: {'center_x': 0.5, 'center_y': .35}
        canvas:
            Color:
                rgb: 13/255,49/255,119/255,1
            Rectangle:
                pos: self.pos
                size: self.size
    MDScrollView:
        do_scroll_x: False
        size_hint: 1, 0.3
        pos_hint: {'center_x': .5, 'center_y': .2}
        MDBoxLayout:
            id: scroll
            orientation: "vertical"
            adaptive_height: True
            MDList:
                id: terminal
'''

class QuestionClassifier(MDApp):
    def on_start(self):
        self.pdfs = []
        self.questions = []
        self.textfields = {}
        
    def build(self):
        return Builder.load_string(kv)

    def open_file_manager(self):
        path = os.getcwd()
        self.file_manager = MDFileManager(
            exit_manager=self.exit_manager,
            select_path=self.select_path,
            ext=['.pdf'],
            search='all'
        )
        self.file_manager.show(path)
       
    def select_path(self, path: str):
        self.exit_manager(1)
        if os.path.isdir(path):
            for file in os.listdir(path):
                self.textfields[path + '/' + file] = textfield.MDTextField(textfield.MDTextFieldHintText(text="Label"), max_height='40dp')
                self.root.ids.container.add_widget(MDListItem(MDListItemLeadingIcon(icon='abacus'),
                                                        MDListItemHeadlineText(text=Path(path + '/' + file).stem, theme_font_name='Custom',
                                                        theme_font_size='Custom',
                                                        font_name="Inter.ttf",
                                                        font_size="14sp",
                                                        theme_text_color='Custom',
                                                        text_color=[0,0,0,1]),
                                                        self.textfields[path + '/' + file]
                                                        ))
                self.pdfs.append(path + '/' + file)
        else:
            self.textfields[path] = textfield.MDTextField(textfield.MDTextFieldHintText(text="Label"), max_height='40dp')
            self.root.ids.container.add_widget(MDListItem(MDListItemLeadingIcon(icon='abacus'),
                                                    MDListItemHeadlineText(text=Path(path).stem, theme_font_name='Custom',
                                                    theme_font_size='Custom',
                                                    font_name="Inter.ttf",
                                                    font_size="14sp",
                                                    theme_text_color='Custom',
                                                    text_color=[0,0,0,1]),
                                                    self.textfields[path]
                                                    ))
            self.pdfs.append(path)
    
    def exit_manager(self, x):
        self.file_manager.close()
    
    def delete_files(self):
        self.root.ids.container.clear_widgets()
        self.root.ids.terminal.clear_widgets()
        self.pdfs.clear()
        self.textfields = {}
        self.questions.clear()

    def test(self):
        if self.root.ids.question.text == '':
            for path in self.pdfs:
                self.questions = pdf_scraping.scrape_questions(type=self.textfields[path].text, questions=self.questions, pdf_path=path)  
            train_df = pd.DataFrame(np.array(self.questions), columns=['text','label1','label2'])
            train_df.to_csv('test.csv')
            get_preds_from_csv(file='test.csv', output='output.csv', model='igcse-physics-all.csv', labels=['electromagnetism','general','nuclear','thermal','waves'])
            pred = pd.read_csv('output.csv')
            keywords = BERTKeywords(train_df['text'], n=10)
            texts = pred['text']
            pred1 = pred['pred1']
            pred2 = [pred_from_keywords('igcse-physics-{}.csv'.format(pred1[i]), keywords[i], labels=list(pd.read_csv('igcse-physics-{}.csv'.format(pred1[i])).columns)[1:]) for i in range(len(keywords))]
            pred['pred2'] = pred2
            predictions = zip(texts.to_list(), pred1.to_list(), pred2, train_df['label1'].to_list(), train_df['label2'].to_list())
            for prediction in predictions:
                self.root.ids.terminal.add_widget(MDListItem(
                                                MDListItemHeadlineText(text=prediction[0], theme_font_name='Custom',
                                                theme_font_size='Custom',
                                                font_name="Inter.ttf",
                                                font_size="14sp",
                                                theme_text_color='Custom',
                                                text_color=[13/255,49/255,119/255,1]), 
                                                MDListItemSupportingText(text=str(prediction[1]) + ', ' + str(prediction[2] + '     {}, {}'.format(str(prediction[3]), str(prediction[4]))), theme_font_name='Custom',
                                                theme_font_size='Custom',
                                                font_name="Inter.ttf",
                                                font_size="13sp",
                                                theme_text_color='Custom',
                                                text_color=[60/255,171/255,250/255,1]), 
                                                ))
            pred.to_csv('output.csv')
            print('Done!')
        else:
            keywords = BERTKeywords(self.root.ids.question.text, n=10)
            pred1 = pred_from_keywords('igcse-physics-all.csv', keywords, ['electromagnetism','general','nuclear','thermal','waves'])
            pred2 = pred_from_keywords('igcse-physics-{}.csv'.format(pred1), keywords, labels=list(pd.read_csv('igcse-physics-{}.csv'.format(pred1)).columns)[1:])
            self.root.ids.terminal.add_widget(MDListItem(
                                                MDListItemHeadlineText(text=self.root.ids.question.text, theme_font_name='Custom',
                                                theme_font_size='Custom',
                                                font_name="Inter.ttf",
                                                font_size="14sp",
                                                theme_text_color='Custom',
                                                text_color=[13/255,49/255,119/255,1]), 
                                                MDListItemSupportingText(text=str(pred1) + ', ' + str(pred2), theme_font_name='Custom',
                                                theme_font_size='Custom',
                                                font_name="Inter.ttf",
                                                font_size="13sp",
                                                theme_text_color='Custom',
                                                text_color=[60/255,171/255,250/255,1]), 
                                                ))
            
QuestionClassifier().run()