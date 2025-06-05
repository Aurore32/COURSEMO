from kivy.lang import Builder
from kivymd.app import MDApp
from kivy.core.window import Window
from kivymd.uix.filemanager import MDFileManager
from kivymd.uix.list import MDListItem, MDListItemHeadlineText, MDListItemTrailingCheckbox, MDListItemLeadingIcon, MDListItemSupportingText
from queue import Queue
from threading import Thread
from kivy.clock import Clock
from kivymd.uix.progressindicator import MDCircularProgressIndicator
from kivy.config import Config
from kivymd.uix.textfield import textfield
import userpaths
import pdf_scraping 
from pathlib import Path
import pandas as pd
import os
import numpy as np
import setfit
from sklearn.model_selection import train_test_split
from datasets import Dataset

from kivymd.uix.dialog import (
    MDDialog,
    MDDialogIcon,
    MDDialogHeadlineText,
    MDDialogSupportingText,
    MDDialogButtonContainer,
    MDDialogContentContainer,
)
from kivymd.uix.list import (
    MDListItem,
    MDListItemLeadingIcon,
    MDListItemSupportingText,
)
from kivymd.uix.button import MDButton, MDButtonText
from kivymd.uix.divider import MDDivider

from kivy.uix.widget import Widget

## Test chain classification
## Train model with slightly more data
## Do other subjects
## Test on A level

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
        pos_hint: {"center_x": .5, "center_y": .59}
        canvas:
            Color:
                rgb: (210/255, 210/255, 210/255, 1)
            RoundedRectangle:
                size: self.size
                pos: self.pos
                radius: [30]
        TextInput:  
            id: model
            hint_text: "Model name..."
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
            id: save
            hint_text: "Save to..."
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
        size_hint: 0.23, 0.06
        pos_hint: {"center_x": .25, "center_y": .41}
        on_press: app.SetFitModel_train()
        MDButtonText:
            text: "Train 1"
            theme_font_name: "Custom"
            theme_font_size: "Custom"
            font_name: "Inter.ttf"
            font_size: "14sp"
            pos_hint: {"center_x": .5, "center_y": .5}
            color: 'white'
    MDButton:
        style: "filled"
        theme_bg_color: "Custom"
        md_bg_color: 0/255,116/255,224/255,1
        theme_width: "Custom"
        theme_height: "Custom"
        size_hint: 0.23, 0.06
        pos_hint: {"center_x": .5, "center_y": .41}
        on_press: app.SetFitModel_train2()
        MDButtonText:
            text: "Train 2"
            theme_font_name: "Custom"
            theme_font_size: "Custom"
            font_name: "Inter.ttf"
            font_size: "14sp"
            pos_hint: {"center_x": .5, "center_y": .5}
            color: 'white'
    MDButton:
        style: "filled"
        theme_bg_color: "Custom"
        md_bg_color: 0/255,116/255,224/255,1
        theme_width: "Custom"
        theme_height: "Custom"
        size_hint: 0.23, 0.06
        pos_hint: {"center_x": .75, "center_y": .41}
        on_press: app.SetFitModel_test()
        MDButtonText:
            text: "Test"
            theme_font_name: "Custom"
            theme_font_size: "Custom"
            font_name: "Inter.ttf"
            font_size: "14sp"
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

    def save_csv(self):
        for path in self.pdfs:
            self.questions = pdf_scraping.scrape_questions(type=self.textfields[path].text, questions=self.questions, pdf_path=path)  
        train_df = pd.DataFrame(np.array(self.questions), columns=['text','label1','label2'])
        train_df.to_csv(self.root.ids.save.text + '.csv')
        return train_df

    def SetFitModel_train(self):
        if os.path.exists(self.root.ids.model.text + '.csv'):
            train_df = pd.read_csv(self.root.ids.model.text + '.csv')
        else:
            train_df = self.save_csv()
        model_name = self.root.ids.model.text
        train_df = train_df[['text','label1']].rename(columns={'text': 'text', 'label1': 'label'})
        model = setfit.SetFitModel.from_pretrained(model_name)
        train, test = train_test_split(train_df, test_size = 0.2)
        train_dataset = Dataset.from_pandas(train)
        test_dataset = Dataset.from_pandas(test)
        model.labels = train_dataset['label']
        train_dataset_sampled = setfit.sample_dataset(dataset=train_dataset, label_column='label', num_samples=10)
        args = setfit.TrainingArguments(num_epochs=(0.5,16),logging_steps=2)
        trainer = setfit.Trainer(model=model, train_dataset=train_dataset_sampled, eval_dataset=test_dataset, args=args)
        trainer.train()
        accuracy = trainer.evaluate(test_dataset)
        model.save_pretrained(self.root.ids.save.text)
        '''        dialog = MDDialog(
        MDDialogIcon(
            icon="check",
            theme_icon_color='Custom',
            icon_color=[0/255,116/255,224/255,1]
        ),
        MDDialogHeadlineText(
            text='Training complete!',
            halign="center",
            theme_font_name='Custom',
            theme_font_size='Custom',
            font_name="Inter.ttf",
            font_size="18sp",
            theme_text_color='Custom',
            text_color='black'
        ),
        MDDialogSupportingText(
            text='Accuracy was {}'.format(accuracy),
            halign="center",
            theme_font_name='Custom',
            theme_font_size='Custom',
            font_name="Inter.ttf",
            font_size="18sp",
            theme_text_color='Custom',
            text_color='black'
        ),
        MDDialogContentContainer(
            MDDivider(color=[0/255,116/255,224/255,1]),
        ),
        MDDialogButtonContainer(
            Widget(),
            MDButton(
                MDButtonText(text="Upload Data", theme_text_color='Custom',
                text_color='green'),
                style="text",
                on_release=model.save_pretrained(self.root.ids.save.text)
            ),
            MDButton(
                MDButtonText(text="Cancel", theme_text_color='Custom',
                text_color='black'),
                style="text",
                on_release=lambda _: dialog.dismiss()
            ),
            spacing="8dp",
        ),
        auto_dismiss=True
        )
        dialog.open()
        except:
        dialog = MDDialog(
        MDDialogIcon(
            icon="refresh",
            theme_icon_color='Custom',
            icon_color=[0/255,116/255,224/255,1]
        ),
        MDDialogHeadlineText(
            text='Model is invalid. Try again?',
            halign="center",
            theme_font_name='Custom',
            theme_font_size='Custom',
            font_name="Inter.ttf",
            font_size="18sp",
            theme_text_color='Custom',
            text_color='black'
        ),
        MDDialogContentContainer(
            MDDivider(color=[0/255,116/255,224/255,1]),
        ),
        MDDialogButtonContainer(
            Widget(),
            MDButton(
                MDButtonText(text="Continue", theme_text_color='Custom',
                text_color='gray'),
                style="text",
                on_release=lambda _: dialog.dismiss()
            ),
            spacing="8dp",
        ),
        auto_dismiss=True
        )
        dialog.open()'''

    def SetFitModel_train2(self):
        if os.path.exists(self.root.ids.save.text + '.csv'):
            train_df = pd.read_csv(self.root.ids.save.text + '.csv')
        else:
            train_df = self.save_csv()
        model_name = self.root.ids.model.text
        labels = train_df['label1'].unique().tolist()
        for label in labels:
            train_df_label = train_df[train_df['label1'] == label][['text','label2']].rename(columns={'text': 'text', 'label2': 'label'})
            model = setfit.SetFitModel.from_pretrained(model_name)
            train, test = train_test_split(train_df_label, test_size = 0.2)
            train_dataset = Dataset.from_pandas(train)
            test_dataset = Dataset.from_pandas(test)
            model.labels = train_dataset['label']
            train_dataset_sampled = setfit.sample_dataset(dataset=train_dataset, label_column='label', num_samples=10)
            args = setfit.TrainingArguments(num_epochs=(2,16))
            trainer = setfit.Trainer(model=model, train_dataset=train_dataset_sampled, args=args)
            trainer.train()
            accuracy = trainer.evaluate(test_dataset)
            model.save_pretrained(self.root.ids.save.text + '_' + label)

    def SetFitModel_test(self):
        if os.path.exists(self.root.ids.save.text + '.csv'):
            train_df = pd.read_csv(self.root.ids.save.text + '.csv')
        else:
            train_df = self.save_csv()
        model_name = self.root.ids.model.text
        model = setfit.SetFitModel.from_pretrained(model_name)
        list = train_df['text'].to_numpy().tolist()
        pred = model.predict(list)
        pred2 = []
        for i in range(len(list)):
            model2 = setfit.SetFitModel.from_pretrained(model_name + '_' + pred[i])
            label2 = model2.predict(list[i])
            pred2.append(label2)
        predictions = zip(list, pred, pred2)
        for prediction in predictions:
            self.root.ids.terminal.add_widget(MDListItem(
                                            MDListItemHeadlineText(text=prediction[0], theme_font_name='Custom',
                                            theme_font_size='Custom',
                                            font_name="Inter.ttf",
                                            font_size="14sp",
                                            theme_text_color='Custom',
                                            text_color=[13/255,49/255,119/255,1]), 
                                            MDListItemSupportingText(text=prediction[1] + ', ' + prediction[2], theme_font_name='Custom',
                                            theme_font_size='Custom',
                                            font_name="Inter.ttf",
                                            font_size="13sp",
                                            theme_text_color='Custom',
                                            text_color=[60/255,171/255,250/255,1]), 
                                            ))
        '''except:
            dialog = MDDialog(
            MDDialogIcon(
                icon="refresh",
                theme_icon_color='Custom',
                icon_color=[0/255,116/255,224/255,1]
            ),
            MDDialogHeadlineText(
                text='Model is invalid. Try again?',
                halign="center",
                theme_font_name='Custom',
                theme_font_size='Custom',
                font_name="Inter.ttf",
                font_size="18sp",
                theme_text_color='Custom',
                text_color='black'
            ),
            MDDialogContentContainer(
                MDDivider(color=[0/255,116/255,224/255,1]),
            ),
            MDDialogButtonContainer(
                Widget(),
                MDButton(
                    MDButtonText(text="Continue", theme_text_color='Custom',
                    text_color='gray'),
                    style="text",
                    on_release=lambda _: dialog.dismiss()
                ),
                spacing="8dp",
            ),
            auto_dismiss=True
            )
            dialog.open()'''
            
QuestionClassifier().run()