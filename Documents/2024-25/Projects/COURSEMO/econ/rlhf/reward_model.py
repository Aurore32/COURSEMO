from openai import OpenAI
import pandas as pd

question = '''Public services – better or worse? 
A government usually has macroeconomic policy aims that it hopes will enable some success in raising the quality of life, or well-being, of the population. In doing this the government also hopes to achieve efficiency in the use of resources.
Efficiency is measured by relating inputs to outputs. Inputs are relatively easy to count: they are financial costs of public services. Output can also be counted, but it is not necessarily a good measure of the outcome. The outcome is much harder to calculate. It is broader and more subjective – how do we assess whether a public service is 'better' or 'worse'?
This is where a knowledge of well-being makes an enormous difference. It could provide a much clearer view of the trade-offs that have to be made in allocating taxpayers' money to public services. Take the example of healthcare. To maximise the impact of expenditure on well-being, the budget may need to be adjusted to give more to mental health services and less to building general hospitals. For older people it could mean giving priority to programmes that would keep them out of hospital. 
A focus on well-being should lead to better outcomes. This is where policymakers need a better understanding of behavioural economics. Governments have established Behavioural Insights teams, or Nudge Units. They have had some success. A small change in the wording of a letter to people who owed tax demonstrated how more behaviourally sensitive language sped up payments. The unit also found that jobseekers were nearly twice as likely to turn up for a job fair if the text message from the job centre used their names, and nearly three times as likely if the person sending the text message added 'good luck'.
Do tax reliefs persuade people to save? No. So enrol them instead automatically in a pensions programme as a default position, with the possibility of opting out. Allowing people to learn from mistakes is good: it reduces dependency on the public sector and helps people make better decisions for themselves. But some errors, such as failing to save anything until you are too old to earn, cannot be reversed. Then an early 'nudge' is justified. It has proved successful in spreading the habit of saving for retirement into groups not persuaded by tax reliefs alone. 
In the long term this, and similar behavioural changes, may well have more influence on well-being than can be represented by concentrating on a monetary calculation of GDP. Other economic indicators could be used to assess this change in well-being.
Source: RSA Issue 1, 2017

The article refers to macroeconomic policy aims. Identify and explain two such macroeconomic policy aims. [4] 
Is there evidence in the article that a knowledge of behavioural economics can help public policy? [4]
The article says that 'efficiency is measured by relating inputs to outputs'. Is this how economic theory states that efficiency is determined? [5]'''

markscheme = '''
a
The article refers to macroeconomic policy aims. Identify and explain two such macroeconomic policy aims.  Any two aims (2) explanation (2) 
4


b
Is there evidence in the article that a knowledge of behavioural economics can help public policy?  • evidence of possible reduction in unemployment • evidence of possible increases in tax payments which could help reduce budget deficit • evidence of persuasion to increase savings might help stop demand inflation 
4


c
The article says that 'efficiency is measured by relating inputs to outputs'. Is this how economic theory states that efficiency is determined?  Partly it is for productive efficiency – cost against output; not really the case for allocative efficiency. 
5


d
The article deals with an improvement in well-being. Discuss whether there are any economic indicators that could be used to assess whether well-being has become better or worse.  Explanation of indicators such as HDI, MEW, GDP per capita; comment on why they are relevant to measuring well-being. 
7
'''

client = OpenAI(api_key="sk-61dd5e4c06c4450893eb429ccf00e0fd", base_url="https://api.deepseek.com")

response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": """
            You are an A-Level economics teacher. You will be given an answer to an A-Level structured question, along with its associated marking scheme and its maximum number of marks.
            You will suggest a mark suitable for this answer. Reference the marking scheme strictly in two ways:
            1. Point-based marking
            - The marking scheme may include sentences or bullet points that indicate appropriate content for the essay.
            - For every point or sentence the mark scheme includes, check if there is content in the given answer that matches that sentence.
            - You will only reward the answer for matching the mark scheme if the answer contains enough detail. For example:
                - Mark scheme: "Some explanation of what is meant by an efficient allocation of resources."
                - Bad answer: "Efficiency is achieved when resources are allocated such that society benefits the most." (do not reward)
                - Good answer: "Whether a market outcome leads to a less efficient allocation of resources can be primarily judged by two types of efficiency: allocative efficiency, which refers to producing at the quantity where the price of the product equals the marginal cost of the product such that every unit of the product produces imparts a marginal social benefit, and productive efficiency, which refers to producing at the lowest possible average total cost." (reward)
            2. Level-based marking
            - The marking scheme may contain level descriptors (Level 1, Level 2, Level 3, Level 4 or L1, L2, L3, L4). Each level corresponds to a range of marks (e.g. 25 marks total, Level 4 (highest level) is 18 to 25 marks). There will be a description of what an answer in each level is expected to achieve. Place the answer in the level most appropriate for it and reference the level when giving the mark.

            Follow these additional rules strictly:
            - A diagram will be indicated in the answer by the text DIAGRAM: (description of diagram). Treat this as if it was an actual diagram and mark accordingly.
            - Assign a numerical mark to the answer based on how many of the marking scheme's points the answer should be rewarded for, as well as its level based on the level descriptors.
            - Your final mark should be a combination of the following factors:
                - How many of the mark scheme points the answer fulfills.
                - How in-depth each of the points are. Look at word count as a heuristic for this.
                - How well the essay follows the structure given by the mark scheme. If the essay does not answer the question properly, or if the mark scheme says to write a balanced response on each side of the question and the essay is one-sided, you should penalize the essay very heavily.
            - Format your response strictly in this style, without any deviation or additional content generated:
            "Total marks: (Your mark)
            Explanation: (Any additional comments or explanations on the mark you may have. Be concise and technical.)"
             """},
                    {
            'role': 'user', 'content': f"""
        Extract: 
        Loyalty and consumer behaviour
        Consumer loyalty can be thought of either in relation to card schemes which offer discounts based on the amount spent with a specific retailer, emotional loyalty where customers are loyal to a particular brand, or as a monopoly loyalty where there is no alternative to the retailer or brand. 
        Retailers recognise that price has an importance in consumers' choice. Many supermarkets use loyalty cards to attract customers with promotions and price reductions available only to those who have a card. 
        By using cards, supermarkets seek a marketing advantage and attempt to build barriers between retailers. Research indicates, however, that although 70% of United Kingdom (UK) consumers hold some kind of loyalty card, only about 10% are loyal to one particular card. 
        Concerns have been raised that, when consumers are collecting points towards a particular goal, the loyalty card schemes may act as a constraint on free competition and prevent switching between brands. It is also thought that the costs of the scheme might be funded through higher prices. Further, the incentives and marketing discounts used by the scheme are often thought to be confusing and make it more difficult for the consumer to compare prices. There are risks for retailers too. It requires substantial investment to run such a scheme – one supermarket put the cost at US$60m a year. 
        But how do loyalty card schemes fit into the context of the way in which consumers make choices? The relative importance of factors that influence consumers when choosing from which shop to buy groceries are one-stop shopping (43%), good service (22%), price (18%), the availability of a coffee shop (12%) and help with packing (6%). Loyalty cards come below these. 
        Companies who have invested in card schemes use them as a means to gather data on customers. When the loyalty card is used the retailer can see what the customer prefers to buy and can direct future offers to their wants in specific promotions both on-line and in the mail. Loyalty schemes are thus sometimes less about loyalty and more about understanding customers' wants. 
        The economic model of consumer behaviour using indifference curves assumes that consumers conduct research and buy products and services in a rational way. However, many purchases are based on habit and consumer research is imprecise. When the consumer has a low involvement in research or a low emotional attachment to the product, there will be little loyalty.
        Source: RSA Journal 1, 2001, page 83

        Question: What evidence is there in the article to suggest that loyalty cards make markets imperfectly competitive? [4]

        Mark Scheme: 
        a
        • Create barriers to competition by reducing switches in demand,• Costs of scheme funded by higher prices• not affordable by smaller firms – high initial costs, $60m for one supermarket,• Causes confusion of information leading to imperfect knowledge. 
        4

        This is a 4-mark question. Remember to answer in continuous prose and follow the guideline for 4-mark questions provided in the system prompt.
        """
        },
        {
            'role': 'assistant', 'content': f"""
        First, the substantial investment required to run the loyalty card scheme (40 million dollars) can be regarded as a high fixed cost to join the market and become competitive, thus becoming a barrier to entry which prevents the market from being perfect. Furthermore, the scheme causes imperfect information by potentially being confusing to consumers, making it difficult for them to reach a rational decision which would have been possible in a perfect market. It is also anti-competitive because it forces consumers to remain devoted to a single seller while collecting loyalty points, which creates differentiation between products beyond price factors and makes the market imperfect.
        """
        },

{
    'role': 'user', 'content': f"""
Question: {question}

Mark Scheme: {markscheme}

Response: {response}
"""
}

],
        stream=False,
        temperature=0.1
    )
print(response.choices[0].message.content)
