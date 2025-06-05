from openai import OpenAI
import pandas as pd

# sk-61dd5e4c06c4450893eb429ccf00e0fd: DeepSeek API key

client = OpenAI(api_key="sk-61dd5e4c06c4450893eb429ccf00e0fd", base_url='https://api.deepseek.com')
responses = []
actual_questions = []
actual_mark_schemes = []

#for i in range(len(questions)):
#    extract = extracts[i]
 #   markscheme = answers[i]
 #   question = questions[i]
df = pd.read_csv('./econ/structured/al_econ_structured_25_mark.csv')
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
                        You are an A-Level economics teacher with full knowledge of the A-Level syllabus. You will be given a 25-mark A-Level structured question, as well as the mark scheme to the question, which should be referenced to generate a response that would obtain full marks according to it. Generate responses in continuous prose (no headings, bullets, or markdown). Follow these rules strictly:
        
        Formatting & Structure
        1. Never use bullets, lists, or headings—write in plain paragraphs.
        2. Stick to economic theory - do not reference real-world examples, even if given in the mark scheme. 
        3. Your essay should follow the following two-part structure:

            Part 1: Analysis. Roughly 800 words.

            - The question will give you an economic statement to assess, consider, evaluate or "agree or disagree" on. This part will be your explanation for why the statement is justified by economic theory.
            - You should begin with a complete economics definition of all the key terms in the question, e.g. "allocative efficienc" or "market economy" or "globalization".
            - Look to the mark scheme for the contents of your essay.
            - The mark scheme will be a list of directions and points with no detail: for example, "The existing equilibrium level of income, output and employment should be identified and explained."
            - Each new line in the mark scheme should be its own paragraph.
            - Your task is to follow the directions in the mark scheme and combine its points together using as much economic detail as possible. 
            - If the mark scheme contains a direction ("Answers should clearly identify and explain why a government might wish to increase the minimum wage"), follow the direction with a focus on these aspects:
                - A detailed definition and explanation of the keywords in the direction ("A minimum wage is a federally mandated floor on wages which employers are legally not allowed to cross.")
                - A complete cause-and-effect chain for everything the direction asks you to "explain" ("A government may wish to increase the minimum wage to counteract inflation. This is because inflation, which is a natural consequence of positive AD shocks accompanying changes in fiscal policy or expansions in the business cycle, causes the erosion of consumer purchasing powers; a wage of 10 dollars would be able to purchase less today than a year ago, for instance. Therefore, the government may want to increase the minimum wage to keep pace with the rate of inflation and ensure that citizens have enough money to support themselves on basic necessities like food and water.")
            - If the mark scheme contains a point ("Sometimes firms will collude and agree to fix a price. This is illegal but it is still used."):
                - Your task is to elaborate on this point as much as possible:
                - Define and explain the keywords in this point if you think it necessary, e.g. "Collusion refers to agreements between firms to fix prices, usually above the market equilibrium in an oligopoly market. This can be in the form of tacit collusion, which does not involve formal agreements, or in the form of a formal agreement and the formation of a cartel."
                - Explain the point with a cause-and-effect chain if applicable ("Firms may choose to collude because of the Prisoner's Dilemma innate in an oligopoly. If a firm sets its price high, all other firms will set their prices low and the firm setting high prices will lose out on all its consumers; this results in a suboptimal Nash equilibrium and firms can counteract this by agreeing to set prices high together.")            - For every point the mark scheme mentions, you will need to include:
                - Precise definitions of terms mentioned in the point, if necessary (e.g., “Allocative efficiency is…”).
       
            Part 2: Evaluation. Roughly 250 words.
            
            - You will need to write a paragraph that evaluates the statement in the question, i.e. consider several factors which may affect whether the statement is true.
            - Possible evaluation points may or may not be given in the mark scheme. If they are not given, think of your own: e.g. the time period in question, the type of economy in question (market or command economy?), the type of good in question, the type of market structure in question, etc. etc.
            - Use conditional language ("This depends on...", "However, if...") for each point.
            - You will need at least 5 evaluative points.
            - Please keep these points separate from the analysis paragraph. Elaborate on each point in depth. Put all the evaluation points in one paragraph and include a conclusion that combines all the points in the end. ("In conclusion, ... (statement) is true given that ..., ... and ... (evaluation points)")
        4. The mark scheme may contain level descriptors (e.g. L4 (18-25), "Level 4: ...", etc.) that describe what a top-marked student is expected to include in their essay. Follow the instructions for the highest level descriptor (L4 (18-25), Level 4, etc.) fully.
                       
        Diagrams
        - If your answer needs to reference a diagram, please:
            - Indicate this via the text (DIAGRAM: (description of what the diagram is showing)).
            - Reference and explain key elements (e.g., "As shown in the monopoly diagram, profit maximization at MR=MC leads to a deadweight loss triangle ABC").
            - Link to analysis (e.g., "The shift from S1 to S2 in the supply diagram illustrates how subsidies reduce production costs").

        Non-Negotiable
        - DO NOT generate any headings, bold text or formatting. Generate your text in plain prose.
        - DO NOT generate any section titles, like "Part 1: Analysis". Plain paragraphs only.
        - DO NOT include any real-world examples in your answer. Economic analysis only.          
        - Your analysis paragraph should be much longer (two times longer) than your evaluation. 
        """},
        {
            'role': 'user', 'content': f"""
        Question:
        'The free market is not the way to achieve a sustainable, efficient use of economic resources. Even the famous economist Adam Smith recognised that there was a need for some government involvement.' Discuss whether government involvement in the economy might overcome the weaknesses of the free market system. [25]


        Mark Scheme:
        Candidates need to answer in terms of market failure and the meaning of the efficient use of
resources and then consider the argument. Market failure can be explained by reference to
possible excessive profits, the need for very large investments, the existence of externalities,
merit goods and of the need for public goods.
 Government intervention can be through regulation, taxation, subsides, or ownership. Price policy
should be to achieve economic efficiency. Economic efficiency should be explained in terms of
productive and allocative efficiency. Neither is necessarily achieved simply because the
government is involved in the production of the good or service.
L4 (18–25) For a thorough explanation of both efficiencies and market failures and a
competent discussion of the role of the government in promoting efficiency
overcoming market failure. A reasoned conclusion should be presented.
L3 (14–17) For a competent explanation with either a comment on both efficiency and market
failure (perhaps concentrating on productive efficiency) or a full explanation of
one but little comment on the other. The role of the government will be discussed
in a limited way but a conclusion should still be presented.
L2 (10–13) For an undeveloped explanation of efficiency/market failure with very little
discussion of the role of the government. Mention of the government will be
descriptive rather than in the form of a discussion related to efficiency/market
failure. It is likely there will be no conclusion.
L1 (1–9) For an answer which shows some knowledge but does not indicate that the
question has been fully grasped, or where the answer contains irrelevancies and
errors of theory.
        """
        },
        {
            'role': 'assistant', 'content': f"""
   A free market system is associated with the capitalist ideology where all resources are privately owned. Hence all economic decisions are made by individual households and firms who act in their own self interest. The essence of a market system is price mechanism, often quoted as "invisible hand". Operating on its own, without government intervention, price mechanism allocates resources through million of decisions taken each day by consumers and businesses. So, it is believed that apparently chaotic system of millions of transactions would not only allocate resources but it would do it efficiently.
Efficiency generally relates to how well an economy uses its scarce resources to satisfy maximum wants of consumers. 

When economists use the term efficiency they usually mean allocative and productive efficiency. Allocative efficiency is when right amount of scarce resources are allocated to produce right products. It occurs when the consumers' valuation (P) of a product equals the cost of resources (marginal cost) used up in its production i.e. p = MC; at that point, all units produced impart a net marginal social benefit, up to the very last unit, where marginal social benefits equal marginal social costs – resulting in a socially optimal outcome. Productive efficiency refers to firms' costs of production and can be applied both to the short run and long run. It exists when producers minimize the wastage of resources i.e when they produce a product using the least possible resources or generating the lowest possible per unit cost. The graph below shows this.

(DIAGRAM: productive efficiency)




The curve AC3 suggests productive efficiency because it indicates the lowest possible cost techniques of production for different levels of output. All other curves above AC3 indicate higher per unit cost therefore they all suggest productive inefficiency. Furthermore, economists use the term technical efficiency when a firm produces at the lowest point (z) on the lowest average cost curve.
The theory of firm predicts that each firm in a perfectly competitive market produces at the lowest point on its LRAC curve and it is not possible for any one firm to reduce its cost any further by altering its output. Every firm in perfect competition is therefore productively efficient. This is shown in the graph below.

(DIAGRAM: cost curves for perfect competition)


It can be analyzed from the graph that perfectly competitive firms maximize profits by equating marginal cost to price. Thus, when perfect competition is the market structure for the whole economy, price is equal to marginal cost in each line of production resulting in allocative efficiency.

On the contrary, a profit maximizing firm operating under imperfect market conditions will maximize profits at an output ( q) where price exceeds marginal cost, hence leading to an inefficient allocation of scarce resources. Also the firm produces at the decreasing part of AC curve that indicates its failure to optimize use of resources.
It is held that even the competitive outcome of markets is not efficient from the point of view of society as a whole. In those cases market system is believed to have failed and it, therefore, results in a loss of economic and social welfare. This is because the benefits that the free-market confers on individuals or businesses carrying out a particular activity diverge from the benefits to the society as a whole.

The existence of externalities, for instance, causes market forces to fail to allocate resource at socially optimum level. An externality occurs whenever actions by firms or consumers impose costs or confer benefits on others that are not involved in the transaction. The essence of the problem created by externalities is that market forces will lead to either too little or too much production. With a positive externality, a competitive free market will produce too little of the good and with a negative externality it produces too much of the good. 

Another important cause of market failure is imperfect information. The reason for this is that party to a transaction can often take an advantage by shifting costs onto the other party. More generally, whenever either party to a transaction lacks information that the other party has, or is deceived by false claims, the outcome of market forces changes and these changes may lead to inefficiency.

Merit and demerit goods are a clear case of imperfect information. Consumers don't perceive quite well how good or bad a particular product is for them: either they do not have the right information or they simply lack some relevant information. The problem is that imperfect information causes market forces to lead to an inappropriate amount of merit and demerit goods being produced or consumed.

A distinct type of market failure exists in case of public goods. A good must possess three distinguishing characteristics to qualify as a public good. It must be non-excludible, non-rival and non-rejectable. These features give rise to the free rider problem; people receiving benefits from a good without contributing to its cost. Thus it develops a situation in which everyone believes that others will take on the burden of paying for goods such as national defense and streetlights. Since the exclusion principle does not apply to these goods, private enterprises have no economic incentive to supply them. So we have goods which yield substantial benefits but to which the market system fails to allocate resources.
The role of the government is to intervene in markets that are not seen to be allocating and using resources in the most efficient manner. Government policy and methods of intervention can be summarized under four broad headings: regulation, financial intervention, direct provision and improving information.

In case of financial intervention indirect taxes such as changes in VAT and excise duties can be used to raise the price of demerit goods and products with negative externalities. Subsidies to consumers will lower the price of merit goods such as grants to students to reduce the internal costs of staying on in fulltime education. In addition to providing the finance it is also possible for a government to take over the production of a good or service, either in whole or in part. State-owned industries are often referred to as nationalized industries such as the electricity, coal mining and railway industries are entirely owned and managed by the state in many countries. It is also very common to find some goods and services being produced by both the state and the private sectors. Education and health care are particularly good examples of these types of service.

Government action can have a role in improving information to help consumers and producers value the 'true' cost and/or benefit of a good or service. Thus a failure to deliver efficient use of resources by the free market system is normally considered as justification for some form of government involvement in the economy. This intervention, however, should be designed to correct market failure with the view to achieve an improvement in economic and social welfare.
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
        new_df['Mark Scheme'] = actual_mark_schemes
        new_df['Response'] = responses
        new_df.to_csv('./econ/distillation/al_econ_25_mark_deepseek_r1.csv')
        '''df = pd.DataFrame([])
        df['Extract'] = extracts[:i+1]
        df['Question'] = questions[:i+1]
        df['Mark Scheme'] = answers[:i+1]
        df['Response'] = responses 
        df.to_csv('al_econ_data_response_deepseek_r1_distill_v2.csv')'''
    except:
        pass
