from openai import OpenAI
import pandas as pd

# sk-61dd5e4c06c4450893eb429ccf00e0fd: DeepSeek API key

client = OpenAI(api_key="sk-61dd5e4c06c4450893eb429ccf00e0fd", base_url='https://api.deepseek.com')
responses = []
actual_questions = []
actual_mark_schemes = []

df = pd.read_csv('./econ/structured/al_econ_structured_dataset_8_mark.csv')
questions = df['Question']
markschemes = df['Mark Scheme']
for i in range(len(questions)):
    question = questions[i]
    markscheme = markschemes[i]
    actual_questions.append(question)
    actual_mark_schemes.append(markscheme)   

    try:
        response = client.chat.completions.create(
                    model="deepseek-reasoner",
                    messages=[
                        {"role": "system", "content": """
                        You are an A-Level economics teacher with full knowledge of the A-Level syllabus. You will be given an 8-mark A-Level structured question, as well as the mark scheme to the question, which should be referenced to generate a response that would obtain full marks according to it. Generate responses in continuous prose (no headings, bullets, or markdown). Follow these rules strictly:
            
            Formatting & Structure
            1. Never use bullets, lists, or headings—write in plain paragraphs.
            2. Stick to economic theory - do not reference real-world examples, even if given in the mark scheme. Your answer should be something a top-scoring student would write in an actual exam.
            3. Your essay should follow the following two-part structure:

                Part 1: Analysis. Roughly 300 words. (Worth 6 marks.)

                - The question will ask you to explain a concept in A-Level economics. Usually, this will be a cause-and-effect statement ("explain how a decrease in interest rates could cause economic growth") or something that requires you to identify two or three different factors ("Explain three causes of unemployment").
                - Do exactly what the question says. Look to the points of the mark scheme for guidance on what points to explain.
                - The mark scheme will usually be in two parts: 
                    - "AO1 Knowledge and Understanding", worth 3 marks, which will ask you to define the key terms in the question and explain what they mean, e.g. "unemployment is the state of being willing and able to work but not having a paid job."
                    - "AO2 Analysis", worth 3 marks, which will ask you to explain the statement in the question, e.g. "One cause of employment is frictional unemployment, which is ..."
                    - If the mark scheme does not explicitly contain these two parts, organize your essay like this based on the points of the mark scheme: an explanation of the keywords using economic language and concepts, and an analysis and explanation of the question's statement.
                - For every point that the mark scheme brings up, you will need to elaborate on it using a full cause-and-effect chain:
                    - An explanation of the key terms of the point,

                Part 2: Evaluation. Roughly 80 words. (Worth 2 marks.)
                
                - You will need to write a paragraph that evaluates the statement in the question, i.e. consider several factors which may affect whether the statement is true.
                - Possible evaluation points will be given in the mark scheme under "AO3 Evaluation".
                - Use conditional language ("This depends on...", "However, if...") for each point.
                - Please keep these points separate from the analysis paragraph. Elaborate on each point in depth. Put all the evaluation points in one paragraph and include a conclusion that combines all the points in the end.
                        
            Diagrams
            - If your answer needs to reference a diagram, please:
                - Indicate this via the text (DIAGRAM: (description of what the diagram is showing)).
                - Reference and explain key elements (e.g., "As shown in the monopoly diagram, profit maximization at MR=MC leads to a deadweight loss triangle ABC").
                - Link to analysis (e.g., "The shift from S1 to S2 in the supply diagram illustrates how subsidies reduce production costs").

            Non-Negotiable
            - DO NOT generate any headings, bold text or formatting. Generate your text in plain prose.
            - DO NOT include any real-world examples in your answer. Economic analysis only.  
            - DO NOT generate any subsection titles, like "Evaluation:" or "Analysis:".
            - Your analysis paragraph should be much longer than your evaluation. Expand on the analysis points from the mark scheme more than on evaluation.             
            """},
            {
            'role': 'user', 'content': f"""
        Question:
        With the use of examples, explain the difference between public goods and merit goods and consider whether markets will always provide enough of both goods. [8]

        Mark Scheme:
        2(a) With the use of examples, explain the difference between public goods
and merit goods and consider whether markets will always provide
enough of both goods.
Follow the point-based marking guidance at the top of this mark scheme and
award:
up to 3 marks for AO1 Knowledge and understanding up to 3 marks for AO2
Analysis
up to 2 marks for AO3 Evaluation.
AO1 Knowledge and Understanding (max 3 marks)
 For the use of accurate examples of BOTH merit goods and public goods
(1).
 For clearly explaining the characteristics of public goods i.e., non-rivalry
and non-excludable (1).
 For clearly explaining the characteristics of merit goods i.e., better than
people realise and under consumed / underproduced due to information
failure (1).
AO2 Analysis (max 3 marks)
 A market economy is less likely to produce enough of both goods due to
the lack of a profit motive unless heavily subsidised.
 A mixed economy may produce enough due to government involvement
and the welfare motive but may be constrained by limits on government
spending.
Note: Analysis should consider at least 2 markets. If only 1 market is analysed,
then 2 marks maximum for analysis and 0 for evaluation.
AO3 Evaluation (max 2 marks)
Offers a valid evaluation of whether enough of both types of good can be
produced in at least 2 markets (1) reaching a valid conclusion (1).
8
AO1 Knowledge and understanding 3
AO2 Analysis 3
AO3 Evaluation 2 
        """
        },
        {
            'role': 'assistant', 'content': """
A merit good refers to a good which carries a positive externality; beyond its benefit provided to the consumer, reflected in the form of the consumer’s demand curve, it also confers an additional, unrealized benefit on either the consumer or on a third-party besides the consumer and the producer. As consumers have insufficient information to realize the true benefit of the good, they undervalue the good and underconsume it as a result of that information failure. For instance, education is a typical case of a merit good: the actual benefit of getting an education is thought to exceed the market price attached to such an education, leading to fewer people receiving an education than is socially optimal.

A public good refers to a good which is both non-excludable, meaning that no consumer can be denied access to the good, and non-rivalrous, meaning that the consumption of the good by one consumer does not deny other consumers the consumption of the same good. For instance, national defense is a public good: as long as you are a citizen of your country you cannot be barred from enjoying the benefits of having a national defense, and you benefiting from national defense does not prevent another person from benefiting from it as well. 

In a market economy, it is very unlikely that enough of both goods will be produced – “enough” referring to the amount which is socially optimal and maximizes societal welfare. Merit goods suffer from information failure and underconsumption by definition; consumer demand for a merit good is less than what it should be because consumers do not realize the true value of the good, resulting in a market equilibrium that is less than the actual social optimum. Public goods will also be underproduced (or not produced at all) due to the existence of the free-rider problem: if firms choose to produce a public good, all consumers can enjoy that public good without being excluded from its benefits through paying a price (non-excludability) or through its consumption by another consumer (non-rivalry), leading to the lack of a profit motive for firms to produce that good because no consumer will pay for it. In other types of markets – a mixed economy, for instance – government intervention in the markets of both goods, which could take the form of subsidies (for merit goods like vaccines or green energy) or direct provision (for public goods like national defense), could mitigate the underconsumption and underproduction of merit and public goods, depending on whether the government is able to spend enough for sufficient provision.

In conclusion, whether merit goods and public goods are provided enough in a market depends on the type of economy the market is situated in – a market economy with no government intervention will lead to the underprovision of both types of goods, while a mixed economy could provide enough through government intervention – as well as, in a mixed economy in particular, the limits placed on government spending. If the market in question is situated within a mixed economy, and if government spending is sufficient to cover the direct provision of public goods, then the goods will be provided enough."""
        },
        {
            'role': 'user', 'content': f"""
        Question: {question}

        Mark Scheme: {markscheme}
        """
        }
        ],
                stream=False,
                temperature=0.6
            )
        print(response.choices[0].message.content)
        responses.append(response.choices[0].message.content)
        new_df = pd.DataFrame([])
        new_df['Question'] = actual_questions
        new_df['Mark Scheme'] = actual_mark_schemes
        new_df['Response'] = responses
        new_df.to_csv('./econ/distillation/al_econ_8_mark_deepseek_r1.csv')
        '''df = pd.DataFrame([])
        df['Extract'] = extracts[:i+1]
        df['Question'] = questions[:i+1]
        df['Mark Scheme'] = answers[:i+1]
        df['Response'] = responses 
        df.to_csv('al_econ_data_response_deepseek_r1_distill_v2.csv')'''
    except:
        pass
