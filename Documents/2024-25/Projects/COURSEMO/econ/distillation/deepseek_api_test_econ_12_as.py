from openai import OpenAI
import pandas as pd

# sk-61dd5e4c06c4450893eb429ccf00e0fd: DeepSeek API key

client = OpenAI(api_key="sk-61dd5e4c06c4450893eb429ccf00e0fd", base_url='https://api.deepseek.com')
responses = []

df = pd.read_csv('./econ/structured/al_econ_structured_dataset_12_mark.csv')
questions = df['Question']
markschemes = df['Mark Scheme']
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
                        You are an A-Level economics teacher with full knowledge of the A-Level syllabus. You will be given an 12-mark AS-Level structured question, as well as the mark scheme to the question, which should be referenced to generate a response that would obtain full marks according to it. Generate responses in continuous prose (no headings, bullets, or markdown). Follow these rules strictly:
        
        Formatting & Structure
        1. Never use bullets, lists, or headings—write in plain paragraphs.
        2. Stick to economic theory - do not reference real-world examples, even if given in the mark scheme. Your answer should be something a top-scoring student would write in an actual exam.
        3. Your essay should follow the following two-part structure:
        - The question will ask you to assess a statement, meaning you have to give a balanced overview on reasons why the statement is true and why it might not be true.
        - Your answer will be in three paragraphs.
        Part 1: First two paragraphs. Analysis. Roughly 600 words. (Worth 8 marks.)

            - The first two paragraphs will be two paragraphs that analyze the given statement in detail. 
                - Look to the mark scheme to determine which two paragraphs you should write. 
                - Some questions will ask you to discuss two policies (e.g. "Assess whether supply-side policy is the best policy for ensuring long-term economic growth"); the mark scheme will say "Analysis of supply-side policy and **one other policy**". In this case, you should write one paragraph on one policy and another paragraph on another policy, like fiscal policy.
                    - If your two paragraphs are each on a different policy, you will need to include an analysis of the advantages and disadvantages of **both** policies (the mark scheme may say "advantages and limitations") in detail. Put the advantages and disadvantages of one policy in one paragraph.
                - Some questions will explicitly ask you to compare two things (e.g. "Assess whether PED is more useful than XED for farmers.") In this case, the first paragraph should be on the first thing mentioned and the second paragraph should be on the second thing mentioned.
                    - If the mark scheme asks you to discuss **both** the benefits and harms of these things, e.g. the mark scheme says "For an analysis of the advantages and limitations of PED", you should include that.
                - Some questions will ask you to discuss the advantages and disadvantages of a single thing (e.g. "Discuss the benefits and harms of a free market system.")
                    - No need to include any other things; one paragraph will be on the benefits of that thing, and the other on its harms.
                - The mark scheme will usually have guidance on what the two paragraphs should be, but you should always write two paragraphs.
            - How to structure each paragraph:
                - A detailed definition and explanation of the keywords for the topic of that paragraph (e.g. "Fiscal policy is a policy measure that involves either a change in tax rates or a change in levels of government spending...")
                    - If the paragraph is about a policy measure, this should include the definition for the policy, the instruments of that policy (e.g. interest rates / exchange rates etc.), and a complete chain of cause-and-effect on how it works, in economic analysis (e.g. lowering interest rates -> cost of borrowing reduced -> more people borrow -> disposable income increases -> consumption increases -> AD shifts to the right)
                    - If the paragraph is about an economic concept (e.g. a market economy, a command economy, PED, XED etc.), include the definition of the concept, a full explanation of what it is (e.g. the formula for PED), and examples to explain it.
                - A detailed discussion of the advantages, disadvantages, or both (if necessary) of the topic being discussed.
                    - Each advantage/disadvantage will include:
                        - A clear statement of the point.
                        - A chain of cause-and-effect explaining the advantage/disadvantage. (e.g. "Fiscal policy is dangerous because it can cause inflation. Fiscal policy shifts AD to the right, which when unaccompanied by a rightward shift in the AS raises both real output and price level. A sustained increase in the price level leads to inflation, which can erode the purchasing powers of consumers over time and reduce consumer confidence in the economy; this will eventually make economic growth unsustainable, as increased output cannot be matched by consumers' purchasing powers, making consumption lag behind.")
                - Diagrams should be drawn accompanying the analysis wherever relevant.
                - DO NOT use any concepts that are not in the AS-Level syllabus. DO NOT draw from real-world evidence or information.
                        
        Part 2: Evaluation. Last paragraph. Roughly 250 words. (Worth 4 marks.)
            
            - You will need to write a paragraph that evaluates the statement in the question, i.e. consider several factors which may affect whether the statement is true.
            - Possible evaluation points will be given in the mark scheme under "AO3 Evaluation".
            - If the mark scheme does not offer any points, think of your own: for instance, consider the time period in question, or the type of good, or the current state of the economy (recession or not), or the budget position of the governments, etc.
            - This is only worth 4 marks, so you will only need to offer 3 to 4 evaluation points plus a conclusion that combines all the points. However, please elaborate on each of the points and explain them in depth with full chains of analysis, as above.
            - Use conditional language ("This depends on...", "However, if...") for each point. 
            - Please keep these points separate from the analysis paragraph. Put all the evaluation points in one paragraph and include a conclusion that combines all the points in the end:
                - "In conclusion, ... is the best policy for ensuring long term economic growth but only if ..., ... and ... (put in your evaluation points)".

        - The mark scheme may also mention level descriptors ("Level 3", "Level 2", etc.) The highest level (e.g. Level 3) will be the one for top marks. Please make sure you have followed the level descriptors for the highest level.

        Diagrams
        - If your answer needs to reference a diagram:
            - Indicate this via the text (DIAGRAM: (description of what the diagram is showing)).
            - Reference and explain key elements (e.g., "As shown in the monopoly diagram, profit maximization at MR=MC leads to a deadweight loss triangle ABC").
            - Link to analysis (e.g., "The shift from S1 to S2 in the supply diagram illustrates how subsidies reduce production costs").

        Non-Negotiable
        - DO NOT generate any headings, bold text or formatting. Generate your text in plain prose.
        - DO NOT include any real-world examples in your answer. Economic analysis only.          
        - Your analysis paragraphs combined should at least be two times longer than your evaluation. Expand on the analysis points from the mark scheme more than on evaluation.             
        """},
        {
            'role': 'user', 'content': f"""
        Question:
        Assess whether supply-side policy is the most effective way to achieve long-run economic growth. [12]
        
        Mark Scheme:
        4(b) Assess whether supply-side policy is the most effective way to achieve
long-run economic growth.
Use Table A: AO1 Knowledge and understanding and AO2 Analysis and Table
B: AO3 Evaluation to mark candidate responses to this question.
AO1 and AO2 out of 8 marks. AO3 out of 4 marks.
Indicative content
Responses may include:
AO1 Knowledge and understanding and AO2 Analysis
 Long-run economic growth is likely to need both growth of AD and AS
 Analysis of the role of supply side policy in achieving long-run economic
growth and its advantages and disadvantages. E.g., supply side policy
such as training and education is more likely to increase AS, but this can
also lead to an increase in AD, for example as better training leads to an
increase in income and therefore an increase in consumer spending. It is
also less likely to result in inflation. However, it is costly and will take time.
 Analysis of the extent to which supply side policy can lead to growth in
both AD and AS. For an explanation that a problem with supply side
policy is that it is less effective in directly increasing the level of AD.
 Analysis of at least one alternative policy e.g., fiscal policy or monetary
policy and how they may achieve long-run economic growth, together
with their advantages and disadvantages e.g., fiscal policy such as a
reduction in income tax is more likely to increase AD and can be relatively
quick. It may also indirectly lead to an increase in AS if more people are
encouraged to enter the workforce. However, if AS does not increase, it
may be inflationary. It may also reduce the amount available for
government spending.
 Analysis of the extent to which these policies can lead to growth in both
AD and AS. For an explanation that a problem with such policies is that
they are less effective in directly increasing the level of AS.
Level 1 responses will be assertive and lacking in explanations / mainly
descriptive and/or or mainly lacking in relevance to the question.
Level 2 responses may contain some inaccuracies and will be one sided.
Analysis will be explained at least in part and will be largely relevant to the
question.
Level 3 responses will consider alternative policies / concepts etc. and will
be balanced. Explanations of points raised will be offered and will be accurate
and relevant to the question.
12

AO3 Evaluation
 Requires an assessment of at least one other government policy to be
compared against supply side policy in achieving long-run economic
growth which leads to
 A justified conclusion as to whether supply side policy is the most
effective way to achieve long-run economic growth or whether a different
policy / combination of policies may be more successful that takes into
account the advantages and disadvantages of such policies in increasing
both AS and AD.
A one-sided response cannot gain any marks for evaluation.
AO1 Knowledge and understanding and AO2 Analysis 8
AO3 Evaluation 4 
        """
        },
        {
            'role': 'assistant', 'content': f"""
On one hand, supply-side policy is certainly instrumental in ensuring long-run economic growth for a country. Supply-side policies encompass a wide range of policies aimed at increasing the productive capacity of the economy, thereby shifting the long-run aggregate supply (LRAS) curve to the right and expanding the economy’s real output. Examples of such policies include subsidies on education, which seek to improve the level of skill and ability present in the workforce through providing more people with education and training, which would improve their productivities. Supply-side policies are an effective driver of long-run economic growth because they lead to sustainable improvements in the quantity and quality of a country’s factors of production, which benefit both producers and consumers; for instance, supply-side policies like education programs or a reduction in the income tax may allow a country’s citizens to benefit from increased levels of disposable income due to gaining skills that would allow them higher salaries, resulting in a simultaneous rise in aggregate demand as well. As such, supply-side policies are also desirable for long-run economic growth because they are less likely to be inflationary than other types of policies: if economic growth derives from a shift in the long-run AS curve or a simultaneous shift of the AS and AD curves to the right, then the potentially inflationary effects of a price-level rise brought by an increase in AD will be counteracted by an increase in AS, effectively preventing economic growth at the expense of eroding consumer purchasing powers. However, it can be argued that supply-side policies carry an innate uncertainty and risk that make it unsuitable for ensuring long-run economic growth. Supply-side policies like immigration reform to allow for an influx of skilled foreign workers have an uncertain effect on the economy, as it is not guaranteed that skilled immigrants will be attracted to the country or that such immigration can occur within a short period of time; in general, from education programs to infrastructure reform, supply-side policies are costly and slow, requiring a great amount of government expenditure for effects that may occur a generation later. Even then, the benefits can be indirect and difficult to observe: for instance, an increase in the literacy rate through supply-side policy may increase disposable incomes which increase AD, but the ultimate effect on AD is hard to guarantee, meaning that supply-side policy cannot influence consumer demand – a crucial component of long-run economic growth – directly.  
 
On the other hand, an alternative to supply-side policies in ensuring long-run economic growth is fiscal policy, aimed at influencing AD through two avenues: an increase in government spending or a reduction in tax rates. Fiscal policy is direct and effective in driving economic growth; for instance, a reduction in income tax rates immediately increases the disposable income of consumers, allowing them to spend more on consumption, which in turn raises real output in the circular flow of income. As such, any form of fiscal policy has immediate and controllable effects on the economy’s real output: a government can control exactly how much it wants to spend or how much it wants to reduce tax rates, which grants it more control over its future economic growth. Moreover, fiscal policy can also stimulate AS simultaneously with AD: for instance, a reduction in income taxes would incentivize workers to seek employment, as the real value of wages have increased from a reduction in the percentage taxed away. This would lead to both an expansion in consumer demand and an expansion in productive capacity, which counteracts inflationary pressures exerted by an increase in AD and results in more sustainable growth. However, it could be argued that fiscal policy is not a sustainable nor a long-term driver of economic growth. Fiscal policy is ultimately a short-run policy: it stimulates consumption or spending in the present and cannot ensure that the economy’s productive capacity, or AS, will cope with this rise in demand. If AS does not rise alongside AD, fiscal policy will exert inflationary pressures on the economy through a rise in the price level; the result of inflation is an erosion of consumers’ purchasing powers, leading to lower levels of consumer confidence and a reduced level of spending in the long-run despite an immediate rise in consumption. Fiscal policy is also an irresponsible policy when the government does not have a sufficient budget surplus to enact government spending: if the spending is discretionary in nature, i.e. it is not spent on education or infrastructure or for any other particular purpose, then a rise in government spending from fiscal policy would limit government expenditures on important programs like healthcare or education, which would potentially damage long-run growth prospects.

Supply-side policies are the most effective way to achieve long-run economic growth only under certain conditions. Supply-side policies can only realize their full potential over the long-term, and is less able to immediately stimulate AD or AS; as such, if an immediate demand-side stimulus is needed to pull the economy out of a recession or increase consumer confidence, fiscal policy would be much more suitable in ensuring a stable economic environment in the short-run as a foundation for long-run economic growth, while supply-side policies will be limited in effectiveness. In addition, only some forms of supply-side policies will result in sustainable long-run growth: any increase in long-run AS (productive capacity) needs to be matched with a rise in AD, and supply-side policies like deregulation will stimulate consumption in the long-run far less than education programs or reductions in income taxes, which increase disposable incomes simultaneously with productive capacities. Finally, whether supply-side policies will be successful versus fiscal policies depend on the political situation at the time: both types of policies are liable to reduce a government’s budget surplus, with supply-side policies like education requiring large amounts of spending and fiscal policies like tax cuts reducing tax revenues, but certain supply-side policies may be politically unpopular depending on the political situation – for instance, immigration reform in times of nationalistic sentiment, or dismantling labor unions in times of anti-corporate sentiment. In conclusion, supply-side policies can be the most effective way to achieve long-run economic growth, but only if there are no pressing economic concerns in the short-term, if the type of policy chosen can stimulate AD and AS simultaneously, and if the political climate is not hostile to the policies used. 
        """
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
        new_df['Mark Scheme'] = actual_markschemes
        new_df['Response'] = responses
        new_df.to_csv('./econ/distillation/al_econ_12_mark_deepseek_r1.csv')
        '''df = pd.DataFrame([])
        df['Extract'] = extracts[:i+1]
        df['Question'] = questions[:i+1]
        df['Mark Scheme'] = answers[:i+1]
        df['Response'] = responses 
        df.to_csv('al_econ_data_response_deepseek_r1_distill_v2.csv')'''
    except:
        pass

