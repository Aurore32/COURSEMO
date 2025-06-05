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
            You are an A-Level economics teacher. You will be given an A-Level data response question (an excerpt, followed by questions on the excerpt), as well as the marking scheme of these questions which tell you how to answer them.
            You will be asked to generate a high-quality response to the question. Write exam answers in **continuous prose** (no headings, bullets, or markdown). Follow these rules strictly:  

1. **Formatting Rules**:  
   - Never use bullet points, lists, headings, or subheadings.  
   - Write in plain, exam-style paragraphs.  
   - Avoid phrases like "In conclusion" or "On the other hand"; instead, use seamless transitions (e.g., "However...", "Ultimately...").  

2. **Answer Structure**:  
   - **Low-mark questions (2-3 marks)**: Direct answers with no analysis (e.g., "A recession is two consecutive quarters of negative GDP growth").  
   - **Mid-mark questions (4-5 marks)**: Explain concepts with 1-2 paragraphs, integrating article quotes (e.g., "The article states 'US$68 billion profits,' which illustrates...").  
   - **High-mark questions (6-8 marks)**:  
     - Analyze both sides using syllabus theory (e.g., "Monopolies restrict output because they face a downward-sloping demand curve, leading to...").  
     - Synthesize article evidence (e.g., "The firms’ 'huge cash reserves' align with theory on barriers to entry").  
     - Conclude organically in the final paragraph. 

3. **Theory & Article Integration**:  
   - Always **name and explain** syllabus concepts (e.g., "allocative inefficiency occurs when...").  
   - For article references, **quote directly** (e.g., "The text mentions 'US$52,000 average salary,' which contrasts with...").  

4. **Non-Negotiable**:  
   - Match response length to marks (8 marks = ~3 paragraphs).  
   - Never separate arguments with labels (e.g., no "Supporting arguments:").  
    - Use the official mark scheme as a guideline to ensure coverage of required concepts, but do not copy its phrasing or structure verbatim. Elaborate heavily on the mark scheme for higher-mark questions.
    - Higher-mark problems should be answered more elaborately, according to the following rules:
            
5. **Elaboration Rules for High-Mark Questions ("Assess" or "Consider", 5-, 6-, 7- and 8-mark):**  
    1. **For every theoretical claim**, explain:  
    - *Cause*: What economic principle or diagram justifies this?  
    - *Effect*: How does this impact stakeholders (consumers, firms, government)?  
    - *Chain of reasoning*: for every theoretical claim, explain the cause-effect chain (e.g., 'X happens because Y, which leads to Z').
    - *Example*: Use syllabus-based examples (e.g., "monopolies restrict output because...").  

    2. **Avoid shorthand explanations**. Replace phrases like "shifts the supply curve" with:  
    *"An excise tax increases production costs, shifting the supply curve leftward from S1 to S2. This internalizes the externality by aligning the marginal private cost (MPC) with the marginal social cost (MSC), reducing output from Q1 to Q2—the socially optimal level."*  

    3. **Always reference diagrams** (even if not drawn):  
    *"As shown by the MSC curve lying above MPC in a negative externality diagram..."*  

    4. **Use A-Level terminology**: Terms like "allocative efficiency," "PED," or "monopsony power" must be **defined and contextualized**.  
    For instance:
    **Mark Scheme**: "Monopolies cause higher prices (1)."  
    **Bad Answer**: "Monopolies cause higher prices."  
    **Good Answer**: "Monopolies raise prices by restricting output. This occurs because, unlike competitive firms, monopolies face no rivals, allowing them to set prices where MR=MC (above MC in a diagram), leading to allocative inefficiency and reduced consumer surplus." 
    **Mark scheme**: (4 marks for support: Monopolies restrict output, higher prices, small firms forced out.)
    **Bad Response** (Copying/Restating):
    "Monopolies restrict output and raise prices, forcing small firms out. This is bad for competition."
    **Good Response** (Elaborating):
    "Monopolies reduce market competition by restricting output—a core A-Level concept. With a downward-sloping demand curve (AR), monopolies maximize profit at MR=MC, setting prices (Pm) above marginal cost (Pc). This creates allocative inefficiency, where prices exceed the competitive equilibrium (Pc), harming consumers. The article’s reference to 'huge cash reserves' exemplifies barriers to entry, a syllabus topic: dominant firms can engage in predatory pricing, driving smaller competitors out, as seen in the text’s mention of investors avoiding rivals. Over time, this reduces market contestability, entrenching monopolistic power."
                      """},
            {"role": "user", "content": r"""Question:
             
            Loyalty and consumer behaviour
            Consumer loyalty can be shown in different ways. It may be through:
            • customer reward schemes (e.g. loyalty cards) that offer discounts based on the amount spent
            with a specific retailer
            • emotional loyalty where customers prefer a particular brand that they always buy
            • monopoly loyalty where there is no alternative to the retailer or brand.
            Supermarkets try to attract customers by using loyalty cards that give promotions and price
            reductions to those who have a card. The cards also help the supermarket build barriers between
            retailers to gain a marketing advantage.
            When consumers are collecting points towards a particular goal, the loyalty card may hinder free
            competition and prevent switching between brands. However, sometimes the discounts used by
            the scheme may be confusing and make it difficult for the consumer to compare prices.
            There are risks for the retailers too. The loyalty cards require substantial investment to run – one
            supermarket put the cost at US$60m a year. These costs could well result in higher prices for the
            consumer.
            But how do loyalty card schemes fit into the context of the way in which consumers make choices?
            A research report found that in one country where 70% of consumers had a loyalty card, only
            about 10% were loyal to one particular card.
            It has been found that when choosing to buy groceries consumers look for one-stop shopping
            (43%), good service (22%), price (18%), the availability of a coffee shop (12%) and help with
            packing (6%). Loyalty cards come below these.
            Companies use loyalty cards to gather data on customers and their buying preferences. They then
            direct future offers to consumer wishes in specific promotions both online and in the mail. Loyalty
            schemes are thus sometimes less about loyalty and more about understanding customers’ wants.
            The economic model of consumer behaviour using indifference curves assumes that consumers
            conduct research and buy products and services in a rational way. Many purchases, however, are
            based on habit and consumer research is imprecise. When the consumer has a low involvement
            in research or a low emotional attachment to the product, there will be little loyalty.
            Source: RSA Journal 

            What evidence is there in the article to suggest that loyalty cards make markets imperfectly
            competitive? [4]
            Explain why a profit-maximising retailer may be interested in the link between consumers’
            loyalty and price elasticity of demand. [4]
            Consider whether there is conflicting evidence in the article about the effectiveness of loyalty
            cards which offer price discounts. [5]
            Assess how the idea of rationality is used in the indifference curve theory of consumer
            behaviour. [7] 

            Mark scheme:
            
            a
            What evidence is there in the article to suggest that loyalty cards make
markets imperfectly competitive?
• Create barriers to competition by reducing switches in demand. (1)
• Costs of scheme funded by higher prices. (1)
• Not affordable by smaller firms – high initial costs, $60m for one
supermarket. (1)
• Causes confusion of information leading to imperfect knowledge. (1)

            b 
            Explain why a profit-maximising retailer might be interested in the link
between consumers’ loyalty and price elasticity of demand.
• Loyalty will mean consumers do not change suppliers, this might create a
lower price elasticity of demand. (1)
• Define price elasticity. (1)
• Explain the link between elasticity and revenue (1); explain the link
between changes in revenue and profits, depending on cost changes. (1)

            c 
            Consider whether there is conflicting evidence in the article about the
effectiveness of loyalty cards which offer price discounts.
Evidence from the article for the effectiveness of using loyalty cards:
• loyalty cards enable retailers to find out about consumer wants and target
promotions (1)
• retailers recognise the importance of price and can offer discounts
through loyalty cards. (1)
In contrast to that the article says:
• price is not the most significant factor in determining choice (1)
• there is no loyalty if buying is based on habit or has low emotional
involvement. (1)
Conclusion: there does seem to be some conflicting evidence. (1)

            d 
            Assess how the idea of rationality is used in the indifference curve
theory of consumer behaviour.
Up to 7 marks for comment on rationality in the use of indifference curves:
• consumers are assumed to make rational choices based on perfect
information (1)
• consumers are assumed not to make choices under uncertainty (1)
• consumer knows combinations between which there is indifference (1)
• this is a rational calculation which remains constant over the curve (1)
• consumer knows satisfaction levels (1)
• consumer knows which combination is preferred to another (1)
• consumer rationally compares satisfaction to price (1)
• consumer buys to achieve maximum satisfaction (1)
• consumer can compare two goods rationally (1)
(max 7)
Guidance:
Max 4 for general description without link to rationality.
              
Remember to write your answer in continuous prose, without headings, bullet points, or subheadings. Use full sentences and paragraphs as a student would in an exam.
             """},
            {"role": "assistant", "content": r"""
             a)	First, the substantial investment required to run the loyalty card scheme (40 million dollars) can be regarded as a high fixed cost to join the market and become competitive, thus becoming a barrier to entry which prevents the market from being perfect. 
             Furthermore, the scheme causes imperfect information by potentially being confusing to consumers, making it difficult for them to reach a rational decision which would have been possible in a perfect market. 
             It is also anti-competitive because it forces consumers to remain devoted to a single seller while collecting loyalty points, which creates differentiation between products beyond price factors and makes the market imperfect.

b)	Price elasticity of demand refers to the percentage change in quantity demanded of a good over the percentage change in its price. 
If consumers’ loyalty is linked to PED in that it makes it more inelastic by increasing the utility consumers derive from consumption, retailers may be able to increase prices to maximize profits, depending on cost changes.
             
c)	On one hand, there is evidence that loyalty card schemes can be occasionally successful. 
70% of all consumers in the UK have loyalty cards in some form, and loyalty cards are a useful metric for firms to gather data on consumers’ wants through monitoring which goods the consumer buys through the card; this allows the firm to influence factors which are important to consumers, such as price (18%) and good service (22%). 
On the other hand, loyalty card schemes can be highly ineffective as suggested by the article. The high start-up costs required for such a scheme (60 million dollars) means that firms may need to raise prices to offset the additional losses, making price reductions from loyalty cards relatively meaningless, and consumers are rarely loyal to only one card (10%), making it an ineffective fosterer of loyalty per se. 
             
d)	Rationality in the context of consumer theory is defined as behavior where the consumer maximizes their own total utility gained from consumption. 
Indifference curves are a representation of all combinations of two goods that the consumer believes would offer them the same level of total welfare. 
In the indifference curve theory, consumers are rational because they can plot out all combinations of two goods that they would prefer equally on the indifference curve, with all such combinations offering the same total utility, which is constant at all points on the curve. 
Furthermore, indifference curve theory assumes that consumers will always prefer a higher utility to a lower utility due to rationality, making them consume where the budget line is tangent to the IC to maximize utility, where they can spend all their income and receive the most welfare possible. 
"""},
        {"role": "user", "content": r"""Question:
Politicians do not usually think it a success to borrow a large amount of money but, in June 2020,
when the economy was approaching a recession, the United Kingdom (UK) borrowed five times
as much as it did in June 2019. This made the government debt only just a little less than the total
size of the economy. The ratio of public sector debt to gross domestic product (GDP) was 99.6%.
It was the highest figure for 60 years. Public sector net borrowing is shown in Fig. 1.

(FIGURE HERE)
         
Fig. 1: Public sector net borrowing in the UK, May 2019 to March 2020
Borrowing is what the government ought to be doing in a recession in order to finance the spending
used to boost aggregate demand. The question in 2020 was how long could the borrowing of such
large amounts continue? There can be problems with a large public sector debt. An expansion
of government borrowing might increase interest rates if investors require more compensation
because they think the risk of holding government bonds has increased. This interest rate rise
would affect the rest of the economy.
There is also a theoretical objection that government borrowing could ‘crowd out’ the private
sector, especially when the economy is close to its productive capacity and unemployment of
resources is low. This would not happen when there are unemployed resources in a recession.
The UK Finance Minister suggested that a high ratio of public sector debt to GDP was justified; it
is the stability of the ratio that is important. This reasoning can be justified. Reducing government
spending in order to reduce borrowing would risk stopping any recovery and make the situation
worse. Economic recovery is necessary to restore economic activity, which in turn will restore the
government’s finances.
However, economists cannot know with any precision what the limits of public sector borrowing
should be. For many years, Japan has maintained a ratio of public sector debt to GDP of over
200% with no difficulty. But what is suitable for Japan may not be suitable in a different type of
economy.
Source: The Times, 24 July 2020
         
State what is meant by ‘a recession’. [3]
Analyse why governments borrow and why it is said that a government should increase
spending in a recession. [5]
Consider what is meant by the phrase ‘government borrowing could ‘crowd out’ the private
sector.’ [6]
Assess whether the evidence in the article makes a conclusive case that the high level of
government borrowing is an effective policy. [6]


            Mark scheme:

a
State what is meant by a recession?
Recession is linked to GDP (1), negative growth of GDP/output (1), over 2
quarters, (1)
3

b
Analyse why governments borrow and why it is said that a government
should increase spending in a recession.
Governments borrow to finance government expenditure not covered by
receipts or taxation (2) Government borrows to finance a deficit (1)
Borrowing would be used to increase government spending/ AMD (1)
this would increase income (1) output (1) employment (1) and through the
multiplier (1) encourage economic growth (1)
5

c
Consider what is meant by the phrase ‘government borrowing could
‘crowd out’ the private sector.
‘Crowding out’ idea is that the growth of government spending will lead to a
transfer of scarce resources, or finance, from the private to the public sector.
(2)
If the government borrows to fund the expenditure it may have to sell
bonds to the private sector (1). Encouraging the private sector to buy bonds
may require higher interest rates (1) or prevent the private sector using the
money for other projects. A rise in interest rates will also make private
borrowing more expensive (1) and crowds-out private investment and
consumption (1)
6

d
Assess whether the evidence in the article makes a conclusive case
that the high level of government borrowing is an effective policy.
Accept any of the following points: reserve a mark for the conclusion.
• Sound because – the economy is in recession so there would be no
crowding out (1)
• the position could be worse if spending were reduced, and government
revenue would be affected further (1)
• high debt is said to be OK if the ratio of debt to GDP is stable (1) but
that debt wasn’t stable, see chart, and it is the highest for 60 years (1)
• Japan doesn’t have a problem (1) but that economy is different (1)
• borrowing may increase interest rates; this would affect the rest of the
economy. High interest rates would discourage investment (1)
Conclusion: Evidence is inconclusive. (1)
6
       
Remember to write your answer in continuous prose, without headings, bullet points, or subheadings. Use full sentences and paragraphs as a student would in an exam.
                """},
            {"role": "assistant", "content": r"""
             a)	A recession is a sustained period of negative GDP growth in an economy over two or more quarters.

             b)	Governments borrow because of their need to finance government spending on programs such as social welfare, healthcare, education, military spending and more. If the government’s projected spending exceeds its revenues, it will run a budget deficit and thus need to borrow money; borrowing money is considered preferable to increasing taxes because of the deflationary effect of taxation and its political risks. In a recession where real output is falling and deflation is likely risin	g, borrowing can have several positive impacts on the economy. First, it is vastly preferable to taxation in correcting a budget deficit, as taxation will further exacerbate falls in real output and price levels, furthering the recession. Borrowing also enables the government to spend more on government programs such as social security, creating a social safety net to those experiencing cyclical unemployment in the recession and also stimulating aggregate demand as to increase both price level and total output. This may enable even further stimulus due to the multiplier effect if the MPC of the economy is fairly high, encouraging economic growth, employment and income. 

             c)	The crowding-out effect refers to a concept in the loanable funds market, where the government’s demand for the scarce resource of loanable funds will ultimately lead to an increased price for such funds which reduce its availability to private firms and individuals, transferring such resources from the private to the public sector. When the government borrows money, it does so by selling bonds on the open market to the public and receiving loanable funds from borrowers. This leads to an increase in the demand for loanable funds, as the same funds could’ve been borrowed by individuals or private firms, which leads to an increase in the real interest rate due to the loanable funds market (diagram). As a result, private investment is crowded out, or reduced, due to higher interest rates increasing the cost to borrow money for investment, leading to an overall reduction in aggregate demand due to the decrease in investment and consumption.

             d)	On one hand, the evidence in the article supports the soundness of government borrowing. First, the article suggests that during a recession, government borrowing is what the government “ought to be doing”; the crowding-out effect is less of a concern due to high unemployment of resources, which the article suggests will happen in a recession. Second, the article also suggests that a high ratio of debt to GDP is not harmful if borrowing leads to higher spending, which “is necessary to restore economic activity” and will in turn restore the government’s finances through taxation and higher incomes. 

            On the other hand, the article does not make a conclusive case as it suggests some potential risks for high borrowing. It is notable that the article only suggests that crowding out and high spending are less of a risk during a recession, implying that in times of economic recovery or booms, government borrowing will be less justified. Furthermore, the article argues that economists will not be able to tell what a reasonable debt-to-GDP ratio is sustainable, as it might differ from economy to economy; Japan has a ratio of “over 200%... without difficulty” while such a ratio “may not be suitable in a different type of economy”. This is particularly concerning given that in the graph shown, the UK’s debt-to-GDP ratio was increasing and fluctuating, and the highest in 60 years, meaning that this stability was not maintained. Therefore, the evidence presented in the article is not conclusive.
"""},            
        {"role": "user", "content": r"""Question:
At its best, capitalism should be all about competition and consumer choice. This leads to innovation,
improved value and quality. It is in contrast to the state, which is about compulsory taxation and
monopoly provision.
Big business is somewhere between capitalism and the state. Unfortunately, companies tend to merge
until they have as much power as a cartel and operate against the public interest. Huge companies
are difficult to manage; full-time executives do not know what is happening – as has been witnessed
by the recent collapse of large banks that required support from huge amounts of taxpayers’ money to
survive.
Other large companies are conglomerates, lacking focus and any real purpose save carrying on for
their own sake. They pay consultancies to devise ‘mission statements’ to give a pretended meaning to
what they stand for. Staff feel detached from the overall undertaking.
Large companies miss interesting opportunities for growth in small markets. They talk about innovation
but do it badly and so compensate by making acquisitions. For example, giant drug companies are
paying large prices to acquire drug development firms because they have lost the confidence to
discover new drugs themselves.
Small and medium-sized companies, by contrast, can adjust and take risks that large companies
cannot. In sectors such as financial services and packaged consumer goods, large companies are
facing competition. Barriers to entry are low, capital is plentiful and technology enables newcomers to
undercut prices of established rivals.
Surveys suggest that young talented people want to work for themselves or smaller companies, not
multinationals beset by internal management politics. New businesses have dynamism and vigour.
They are more productive and more innovative. Fragmented markets and ever more diverse tastes
of consumers threaten the dominance of the old brands. Economies of scale are a myth in many
industries, especially newer sectors such as technology and services.
There is a belief that companies need to be huge to trade globally and that there are efficiencies from
scale when trading internationally. These are exaggerated in the modern era. Small and medium-sized
companies can be flexible and adaptable. With an improving economy and falling oil prices, smaller
businesses have a great opportunity for improving profits and maintaining prices. Large firms remain
uniform with dull, identical brands everywhere. It’s time for the craft industries, the food specialists, the
new banks and the break-up of the dominant corporations.
Source: Sunday Times, 12 April 2015
         
Explain what is meant by a positive statement and a normative statement and consider
whether the first paragraph should be classified as positive or normative. [4]
Briefly describe what is meant by a cartel and explain why it is said to be against the public
interest. [4]
Analyse what is meant by the statement that ‘there are efficiencies from scale when trading
internationally’. [4]
The article is headed ‘Big is bad’. Discuss whether there is sufficient evidence provided to
support that statement. [8]
         
            Mark scheme:
a
positive: objective, can be tested. Normative: value judgement. 2 First
paragraph has both. Use of ‘should’ – normative, others positive. 2
4

b         
group acting together, fixes output, fixes prices, prevents competition, 4

c         
comment on meaning of economies of scale and seeming relevance to large
international markets.
4

d         
some evidence to support about consultancies, lack of innovation,
acquisitions not innovation, managerial problems (e.g. of banks), tendency to
cartels – but no figures, or details. There is also evidence about niche
markets, economies of scale, international trade – but no details or success
rates, price changes, costs, whether profits have improved. no source for
surveys re young people. Oversimplification of role of state. Some normative
words: dull identical brands, innovate badly, give pretended meaning to their
aim. Article presents negative side of large business and beneficial side of
smaller businesses. Not a balanced view. Max 7 no conclusion 
8
          
Remember to write your answer in continuous prose, without headings, bullet points, or subheadings. Use full sentences and paragraphs as a student would in an exam.
             """},
            {"role": "assistant", "content": r"""
            a)	A positive statement is objective and factually testable, while normative statements are subjective value judgments. In the first paragraph, the statement that capitalism “at its best should be all about competition and consumer choice”; the usage of “should” indicates a value judgment of what things ought to be, which is a normative statement. However, the statement that the state is about “compulsory tax” is truthful and is thus a positive statement; capitalism leading to “innovation and improve quality” is also factually testable. Both types are present in the paragraph.

             b)	A cartel is a formalized collection of firms, usually found in a monopoly, bound by a legal agreement to restrict output to collectively drive up prices and eearn more profits. Cartels are said to be against the public interest because they impose quotas on individual firm production and reduces output below allocative efficiency. Furthermore, reducing supply has the effect of increasing price to consumers, harming their interests. Cartels also hold an excessive level of market power which has the risk of monopolizing the market in an anti-competitive way, earning profits at the expense of consumers.

             c)	Efficiencies from scale refers to the economies of scale firms benefit from, where their average total costs in the long run decrease as a result of their increased output. This is for several reasons, including technological (increased scale and profits lend themselves to re-investment into research and development and investment into more efficient capital, improving productivity), managerial (reduced costs from skilled management), and risk-bearing (diversification of goods allows for more risk-taking). Firms that trade internationally and operate in multiple countries have more opportunities to gain economies of scale due to their larger size, which allows them to out-compete competitors both domestically and abroad.

             d)	On one hand, the article provides ample evidence for the negative impacts of large firms. It is stated that often, large firms expand through mergers that operate like cartels, which harm the public interest through inflating price, reducing output and monopolizing the market. This also leads to managerial diseconomies of scale and diminishing marginal returns, as evidenced by big banks failing due to poor management and excessive size to manage; it is further evidenced by employees feeling “detached from the overall undertaking” and large firms “lacking focus”, which reduces efficiency significantly and drives younger people into wanting to work for smaller businesses with “dynamism and vigor” and more productivity. Finally, evidence is given that large firms lack the innovative ability of small firms due to their fear of risk-taking and inability to enter small, innovative markets.
    On the other hand, evidence is also given that large firms could potentially have their advantages. It is stated that large firms have advantages when trading internationally due to their economies of scale, which small firms do not benefit from as much; this can be technical, managerial or risk-taking. The evidence provided that large firms are less supported by the young, less profitable and less productive is not evidenced by statistics or sources, making it less credible; focus is given instead to normative statements such as “pretended meaning” and “economies of scale are a myth in many industries”.
    Overall, though the article provides evidence and examples supporting the claim that large firms are worse-off, its sources and claims are not credible and its arguments are rarely balanced, meaning that the conclusion remains unclear.
"""},
        {"role": "user", "content": r"""Question:
The impact of economic concentration
By locating closely together manufacturers can often gain economic benefits. The makers of
industrial machines save on transport and communication costs when they are close to the firms
that supply their raw materials and components. A group of similar manufacturers attracts a large
number of workers with similar skills. Educational and training facilities in the area will develop
courses and research skills that support the local industry. Similar benefits also occur in service
industries, for example with the concentration of information technology in Silicon Valley and
finance in New York, Hong Kong and London, where groups of accountants, merchant bankers
and specialist lawyers in mergers and acquisitions all exist in close proximity.
The scope for concentration depends on the size of the market. The recent trend towards
globalisation has increased the size of the available market. This has allowed some internet
retailing firms to overcome competition from other countries. But the overall impact has been
the delivery of a wide range of goods and services at low prices that have benefited consumers
everywhere. This suggests that globalisation has not been a ‘zero-sum’ game.
But concentration has not always been beneficial. Silicon Valley has significant traffic congestion
and many local people are priced out of the housing market. Seven of the ten most polluted cities
in China are in Hebei Province, a centre for the iron and steel industry. This concentration has
caused large amounts of air, water and land pollution. The competing demand for a limited number
of workers has also led to wage increases.
Geographical concentration of an industry can lead to problems when demand for the product
changes. This has been evident in the United States (US) where the city of Detroit has experienced
unemployment rates as high as 25% (see Fig. 1.1) and a fall in its population by over 60%
because of the decline in Detroit’s motor industry caused by vehicle imports and the movement of
production to US states where trade unions are not as strong.
Source: The Economist, 21 October 2017
         
Explain the meaning of zero-sum in game theory. [2]
Use the information to explain how external economies of scale develop. Show, with the aid
of a diagram, how this would affect a firm’s average costs. [5]
There is a fall in demand for the product of a geographically concentrated industry. Analyse
the economic implications of this for the area where the industry is located. [6]
Consider how the negative externalities from concentrated industrial production can be
reduced. [7]
         
            Mark scheme:
a
Explain the meaning of zero-sum in game theory.
The total benefit is fixed (1) and one party can only gain if the other party loses
(1)
2

b         
Use the evidence to explain how external economies of scale develop.
Show, with the aid of a diagram, how this would affect a firm’s average
costs.
Common supply of raw materials/components/pool of skilled labour.
Development of ancillary services such as education/legal.
These lead to external economies of scale which reduce long run average cost
for all companies operating in the area/region.
2 + 2 + 1 for diagram
5

c
There is a fall in demand for the product of a concentrated industry.
Analyse the economic implications of this for the area where the industry
is located.
A fall in demand will be an example of structural unemployment, falling
incomes, a local negative multiplier effect, falling house prices, movement of
population.
An analysis of 3 effects x 2
6

d         
Consider how the negative externalities from concentrated industrial
production can be reduced.
Explanation of negative externalities with reference to passage 1
Consideration of the role of 2 policies from
taxation/subsidy/legislation/education 3 + 3
         
Remember to write your answer in continuous prose, without headings, bullet points, or subheadings. Use full sentences and paragraphs as a student would in an exam.
              """},
            {"role": "assistant", "content": r"""
           a)	Zero-sum in game theory refers to a situation where the gains of one firm/many firms are equal to the losses incurred by other firms; no member of the game can benefit without being at the expense of another member of the game losing out. The net change in benefit is zero in any such case.
b)	External economies of scale refer to decreases in a firm’s LRATC through a general increase in the size of the market for a firm. As the article states, ways in which this can occur include locating firms closely together to decrease transportation and communication costs due to proximity to suppliers, encouraging development of education and training in the area to support the industry, and attracting skilled labor to relocate to the area. This is shown by a shift downward in an average total cost curve in the LR:
c)	When demand falls in a geographically concentrated industry, the economic consequences can be severe to the geographical region. First, mass unemployment may occur because many firms in the industry may reach the shutdown point where P < AVC, leading to labor no longer being employed; the information offers an example of a 25% unemployment rate in Detroit, which was sustained over several years to upwards of 15%, far more than the national average. There may also be a significant decrease in the population of the area because of lowered job opportunities and lower living standards; the income per capita of the region will almost surely decrease. Demand for housing in the area will decrease.
d)	Negative externalities are negative impacts on a third-party that the production or consumption of a good may cause, e.g. air pollution that the article mentions. First, such externalities can be reduced through using government intervention, such as through excise/indirect taxes on production. An excise tax on the product shifts supply to the left and thus allows the marginal private cost curve to coincide with the marginal social cost curve which reduces the production of the product and corrects the negative externality. Specifically, the government may consider a carbon tax, which punishes firms the more carbon dioxide they emit, allowing them to retain their profits if they do not pollute and thus eliminating negative externalities in production. Alternative sources of intervention, such as maximum prices to make the product less profitable and reduce its supply, can also be used. (diagram). Furthermore, a method such as applying Coase’s theorem through pollution permits can be used; a government may charge a certain price for a pollution permit which allows firms to pollute and penalize firms which pollute without permits, and the additional cost incurred by the firm through the pollution permit will increase its costs and force it to lower production, thus decreasing externalities.
"""},
{
    'role': 'user', 'content': f"""
Question: {question}

Mark Scheme: {markscheme}

Write a response in continuous prose (no bullets/headings). Expand all mark scheme points with A-Level theory and article quotes. Do not copy the mark scheme.
"""
}

],
        stream=False,
        temperature=0.3
    )
print(response.choices[0].message.content)
