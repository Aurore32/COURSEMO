from openai import OpenAI
import pandas as pd

dataset = pd.read_csv('al_econ_data_response_dataset_split.csv')
questions = dataset['Question'][155:].tolist()
answers =  dataset['Mark Scheme'][155:].tolist()
extracts = dataset['Extract'][155:].tolist()
client = OpenAI(api_key="sk-61dd5e4c06c4450893eb429ccf00e0fd", base_url="https://api.deepseek.com")
responses = []


for i in range(len(questions)):
    extract = extracts[i]
    markscheme = answers[i]
    question = questions[i]
    try:
        response = client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {"role": "system", "content": """
        You are an A-Level economics teacher. You will be given an A-Level data response question, which includes an economics article or extract and a question concerning it, as well as the mark scheme to the question, which should be referenced to generate a response that would obtain full marks according to it. Generate responses to data-response questions in continuous prose (no headings, bullets, or markdown). Follow these rules strictly:
        Formatting & Structure
        1.	Never use bullets, lists, or headings—write in plain paragraphs.
        2.	Match length to marks:
        •	2-3 marks: 1–2 sentences (direct answer, no analysis).
        •	4-5 marks: 1–2 paragraphs (explain concepts + 1-2 article quotes).
        •	6-8 marks: 3–4 paragraphs (analyze both sides, use theory + evidence).
        •   Questions asking to "state": simple statement of point.
        •   Questions asking to "explain": 1 statement of a point + an explanation of that point based on A-Level economics knowledge.
        •   Questions asking to "consider", "discuss", or "assess": full analysis of valid points on both sides of the argument + conclusion that reaches a final judgement.
        Theory & Article Integration
        1.	Name and explain syllabus concepts (e.g., “Monopolies restrict output due to their downward-sloping AR curve, leading to allocative inefficiency where P > MC”).
        2.	Quote the article directly, if asked to do so in the question (e.g., “The ‘US$68 billion profits’ reflect supernormal profits”).
        3.	Reference diagrams in text (e.g., “In a negative externality diagram, MSC lies above MPC”).
        4.  You should quote the article whenever necessary, but ensure that if the question does not ask you about the article, focus predominantly on economic theory.
        High-Mark Elaboration (6-8 marks)
        For every theoretical claim, include:
        1.	Cause: “X occurs because...” (e.g., “Prices rise because monopolies set output where MR=MC”).
        2.	Effect: “This leads to...” (e.g., “Higher prices reduce consumer surplus”).
        3.	Chain of reasoning: “As a result...” (e.g., “Predatory pricing deters competitors, reducing contestability”).
        4.	Syllabus example: “This aligns with the concept of...” (e.g., “Barriers to entry”).
        Non-Negotiables
        •	Never copy/paraphrase the mark scheme. Use it only to identify required concepts.
        •	Expand all mark scheme points:
        o	Turn “monopolies raise prices” → “Monopolies raise prices by restricting output (Qm < Qc) due to profit maximization at MR=MC, creating deadweight loss (DWL) as shown in monopoly diagrams.”
        •	Avoid shorthand: Replace “shifts supply” → “An excise tax shifts S left from S1 to S2, raising equilibrium price from P1 to P2.”
        Examples of Good vs. Bad Responses
        •	Mark Scheme Snippet: “Monopolies cause higher prices (1).”
        o	Bad: “Monopolies cause higher prices.”
        o	Good: “Monopolies raise prices by restricting output. With a downward-sloping AR curve, they maximize profit at MR=MC (Qm), setting Pm > MC. This creates allocative inefficiency (P > MC), reducing consumer surplus—a core A-Level concept.”
        •	Mark Scheme Snippet: “Barriers to entry deter competitors (1).”
        o	Bad: “Large firms use barriers to entry.”
        o	Good: “Dominant firms exploit barriers to entry, like ‘huge cash reserves’ (article quote), to fund predatory pricing. This deters new entrants, as per the syllabus’s contestable markets theory, entrenching monopoly power.”
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

        Question: Analyse why a profit maximising retailer might be interested in the link between utility, consumers' loyalty and price elasticity of demand. [4]

        Mark Scheme: 
        b
        • Loyalty will mean consumers do not change suppliers, this might create a lower price elasticity of demand, (1)• Explain price elasticity (1)• With lower price elasticity can subsequently raise prices with resulting higher profits – depending on cost changes (2) 
        4

        This is a 4-mark question. Remember to answer in continuous prose and follow the guideline for 4-mark questions provided in the system prompt.
        """
        },
        {
            'role': 'assistant', 'content': f"""
        Price elasticity of demand refers to the percentage change in quantity demanded of a good over the percentage change in its price. If consumers’ loyalty is linked to PED in that it makes it more inelastic by increasing the utility consumers derive from consumption, retailers may be able to increase prices to maximize profits, depending on cost changes.
        """
        },
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

        Question: Consider whether there is conflicting evidence in the article about the effectiveness of loyalty cards which offer price discounts. [5]

        Mark Scheme: 
        c
        Evidence for the effectiveness of cards:• Cards enable retailers to find out about consumer wants and target promotions;• Article also says retailers recognise the importance of price and can offer discounts through cards. (2)In contrast to that it also says• Price is not most significant factor in determining choice;• Also there is no loyalty if buying is based on habit, or has low emotional involvement. (2)Conclusion: seems to be some conflict. (1) 
        5

        This is a 5-mark question. Remember to answer in continuous prose and follow the guideline for 5-mark questions provided in the system prompt.
        """
        },
        {
            'role': 'assistant', 'content': f"""
        On one hand, there is evidence that loyalty card schemes can be occasionally successful. 70% of all consumers in the UK have loyalty cards in some form, and loyalty cards are a useful metric for firms to gather data on consumers’ wants through monitoring which goods the consumer buys through the card; this allows the firm to influence factors which are important to consumers, such as price (18%) and good service (22%). On the other hand, loyalty card schemes can be highly ineffective as suggested by the article. The high start-up costs required for such a scheme (60 million dollars) means that firms may need to raise prices to offset the additional losses, making price reductions from loyalty cards relatively meaningless, and consumers are rarely loyal to only one card (10%), making it an ineffective fosterer of loyalty per se. 
        """
        },
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

        Question: Discuss how the idea of rationality is used in the indifference curve theory of consumer behaviour. [7]

        Mark Scheme: 
        d
        Explanation of indifference curves. General description without link to rationality max 4 Consumer knows combinations between which there is indifference, this is a rational calculation which remains constant over the curve,Consumer knows satisfaction levels and which combination is preferred to another,Consumer rationally compares satisfaction to price, and buys to achieve maximum satisfactionConsumer can compare two goods using rational deductions. 
        7

        This is a 7-mark question. Remember to answer in continuous prose and follow the guideline for 7-mark questions provided in the system prompt.
        """
        },
        {
            'role': 'assistant', 'content': f"""
        Rationality in the context of consumer theory is defined as behavior where the consumer maximizes their own total utility gained from consumption. Indifference curves are a representation of all combinations of two goods that the consumer believes would offer them the same level of total welfare. In the indifference curve theory, consumers are rational because they can plot out all combinations of two goods that they would prefer equally on the indifference curve, with all such combinations offering the same total utility, which is constant at all points on the curve. Furthermore, indifference curve theory assumes that consumers will always prefer a higher utility to a lower utility due to rationality, making them consume where the budget line is tangent to the IC to maximize utility, where they can spend all their income and receive the most welfare possible. 
        """
        },
        {
            'role': 'user', 'content': f"""
        Extract: 
        Increasing public sector debt is a good policy
        Politicians do not usually think it a success to borrow a large amount of money but, in June 2020, when the economy was approaching a recession, the United Kingdom (UK) borrowed five times as much as it did in June 2019. This made the government debt only just a little less than the total size of the economy. The ratio of public sector debt to gross domestic product (GDP) was 99.6%. It was the highest figure for 60 years. Public sector net borrowing is shown in Fig. 1.
        (FIGURE)
        Borrowing is what the government ought to be doing in a recession in order to finance the spending used to boost aggregate demand. The question in 2020 was how long could the borrowing of such large amounts continue? There can be problems with a large public sector debt. An expansion of government borrowing might increase interest rates if investors require more compensation because they think the risk of holding government bonds has increased. This interest rate rise would affect the rest of the economy.
        There is also a theoretical objection that government borrowing could 'crowd out' the private sector, especially when the economy is close to its productive capacity and unemployment of resources is low. This would not happen when there are unemployed resources in a recession. 
        The UK Finance Minister suggested that a high ratio of public sector debt to GDP was justified; it is the stability of the ratio that is important. This reasoning can be justified. Reducing government spending in order to reduce borrowing would risk stopping any recovery and make the situation worse. Economic recovery is necessary to restore economic activity, which in turn will restore the government's finances.
        However, economists cannot know with any precision what the limits of public sector borrowing should be. For many years, Japan has maintained a ratio of public sector debt to GDP of over 200% with no difficulty. But what is suitable for Japan may not be suitable in a different type of economy.
        Source: The Times, 24 July 2020

        Question: State what is meant by ‘a recession’. [3]

        Mark Scheme: 
        a

        State what is meant by a recession?
        Recession is linked to GDP (1), negative growth of GDP/output (1), over 2 quarters, (1)

        3


        This is a 3-mark question. Remember to answer in continuous prose and follow the guideline for 3-mark questions provided in the system prompt.
        """
        },
        {
            'role': 'assistant', 'content': f"""
        A recession is a sustained period of negative GDP growth in an economy over two or more quarters.
        """
        },
        {
            'role': 'user', 'content': f"""
        Extract: 
        Increasing public sector debt is a good policy
        Politicians do not usually think it a success to borrow a large amount of money but, in June 2020, when the economy was approaching a recession, the United Kingdom (UK) borrowed five times as much as it did in June 2019. This made the government debt only just a little less than the total size of the economy. The ratio of public sector debt to gross domestic product (GDP) was 99.6%. It was the highest figure for 60 years. Public sector net borrowing is shown in Fig. 1.
        (FIGURE)
        Borrowing is what the government ought to be doing in a recession in order to finance the spending used to boost aggregate demand. The question in 2020 was how long could the borrowing of such large amounts continue? There can be problems with a large public sector debt. An expansion of government borrowing might increase interest rates if investors require more compensation because they think the risk of holding government bonds has increased. This interest rate rise would affect the rest of the economy.
        There is also a theoretical objection that government borrowing could 'crowd out' the private sector, especially when the economy is close to its productive capacity and unemployment of resources is low. This would not happen when there are unemployed resources in a recession. 
        The UK Finance Minister suggested that a high ratio of public sector debt to GDP was justified; it is the stability of the ratio that is important. This reasoning can be justified. Reducing government spending in order to reduce borrowing would risk stopping any recovery and make the situation worse. Economic recovery is necessary to restore economic activity, which in turn will restore the government's finances.
        However, economists cannot know with any precision what the limits of public sector borrowing should be. For many years, Japan has maintained a ratio of public sector debt to GDP of over 200% with no difficulty. But what is suitable for Japan may not be suitable in a different type of economy.
        Source: The Times, 24 July 2020

        Question: Analyse why governments borrow and why it is said that a government should increase spending in a recession. [5]

        Mark Scheme: 
        b

        Analyse why governments borrow and why it is said that a government should increase spending in a recession.
        Governments borrow to finance government expenditure not covered by receipts or taxation (2) Government borrows to finance a deficit (1)
        Borrowing would be used to increase government spending/ AMD (1) this would increase income (1) output (1) employment (1) and through the multiplier (1) encourage economic growth (1)

        5


        This is a 5-mark question. Remember to answer in continuous prose and follow the guideline for 5-mark questions provided in the system prompt.
        """
        },
        {
            'role': 'assistant', 'content': f"""
        Governments borrow because of their need to finance government spending on programs such as social welfare, healthcare, education, military spending and more. If the government’s projected spending exceeds its revenues, it will run a budget deficit and thus need to borrow money; borrowing money is considered preferable to increasing taxes because of the deflationary effect of taxation and its political risks. In a recession where real output is falling and deflation is likely rising, borrowing can have several positive impacts on the economy. First, it is vastly preferable to taxation in correcting a budget deficit, as taxation will further exacerbate falls in real output and price levels, furthering the recession. Borrowing also enables the government to spend more on government programs such as social security, creating a social safety net to those experiencing cyclical unemployment in the recession and also stimulating aggregate demand as to increase both price level and total output. This may enable even further stimulus due to the multiplier effect if the MPC of the economy is fairly high, encouraging economic growth, employment and income. """
        },
        {
            'role': 'user', 'content': f"""
        Extract: 
        Increasing public sector debt is a good policy
        Politicians do not usually think it a success to borrow a large amount of money but, in June 2020, when the economy was approaching a recession, the United Kingdom (UK) borrowed five times as much as it did in June 2019. This made the government debt only just a little less than the total size of the economy. The ratio of public sector debt to gross domestic product (GDP) was 99.6%. It was the highest figure for 60 years. Public sector net borrowing is shown in Fig. 1.
        (FIGURE)
        Borrowing is what the government ought to be doing in a recession in order to finance the spending used to boost aggregate demand. The question in 2020 was how long could the borrowing of such large amounts continue? There can be problems with a large public sector debt. An expansion of government borrowing might increase interest rates if investors require more compensation because they think the risk of holding government bonds has increased. This interest rate rise would affect the rest of the economy.
        There is also a theoretical objection that government borrowing could 'crowd out' the private sector, especially when the economy is close to its productive capacity and unemployment of resources is low. This would not happen when there are unemployed resources in a recession. 
        The UK Finance Minister suggested that a high ratio of public sector debt to GDP was justified; it is the stability of the ratio that is important. This reasoning can be justified. Reducing government spending in order to reduce borrowing would risk stopping any recovery and make the situation worse. Economic recovery is necessary to restore economic activity, which in turn will restore the government's finances.
        However, economists cannot know with any precision what the limits of public sector borrowing should be. For many years, Japan has maintained a ratio of public sector debt to GDP of over 200% with no difficulty. But what is suitable for Japan may not be suitable in a different type of economy.
        Source: The Times, 24 July 2020

        Question: Consider what is meant by the phrase 'government borrowing could 'crowd out' the private sector.' [6]

        Mark Scheme: 
        c

        Consider what is meant by the phrase 'government borrowing could 'crowd out' the private sector.
        'Crowding out' idea is that the growth of government spending will lead to a transfer of scarce resources, or finance, from the private to the public sector. (2) 
        If the government borrows to fund the expenditure it may have to sell bonds to the private sector (1). Encouraging the private sector to buy bonds may require higher interest rates (1) or prevent the private sector using the money for other projects. A rise in interest rates will also make private borrowing more expensive (1) and crowds-out private investment and consumption (1)

        6

        This is a 6-mark question. Remember to answer in continuous prose and follow the guideline for 6-mark questions provided in the system prompt.
        """
        },
        {
            'role': 'assistant', 'content': f"""
        The crowding-out effect refers to a concept in the loanable funds market, where the government’s demand for the scarce resource of loanable funds will ultimately lead to an increased price for such funds which reduce its availability to private firms and individuals, transferring such resources from the private to the public sector. When the government borrows money, it does so by selling bonds on the open market to the public and receiving loanable funds from borrowers. This leads to an increase in the demand for loanable funds, as the same funds could’ve been borrowed by individuals or private firms, which leads to an increase in the real interest rate due to the loanable funds market (diagram). As a result, private investment is crowded out, or reduced, due to higher interest rates increasing the cost to borrow money for investment, leading to an overall reduction in aggregate demand due to the decrease in investment and consumption.
        """
        },
        {
            'role': 'user', 'content': f"""
        Extract: 
        Big is bad
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

        Question: The article is headed ‘Big is bad’. Discuss whether there is sufficient evidence provided to support that statement. [8]

        Mark Scheme: 
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


        This is a 8-mark question. Remember to answer in continuous prose and follow the guideline for 8-mark questions provided in the system prompt.
        """
        },
        {
            'role': 'assistant', 'content': f"""
        d)	On one hand, the article provides ample evidence for the negative impacts of large firms. It is stated that often, large firms expand through mergers that operate like cartels, which harm the public interest through inflating price, reducing output and monopolizing the market. This also leads to managerial diseconomies of scale and diminishing marginal returns, as evidenced by big banks failing due to poor management and excessive size to manage; it is further evidenced by employees feeling “detached from the overall undertaking” and large firms “lacking focus”, which reduces efficiency significantly and drives younger people into wanting to work for smaller businesses with “dynamism and vigor” and more productivity. Finally, evidence is given that large firms lack the innovative ability of small firms due to their fear of risk-taking and inability to enter small, innovative markets.
        On the other hand, evidence is also given that large firms could potentially have their advantages. It is stated that large firms have advantages when trading internationally due to their economies of scale, which small firms do not benefit from as much; this can be technical, managerial or risk-taking. The evidence provided that large firms are less supported by the young, less profitable and less productive is not evidenced by statistics or sources, making it less credible; focus is given instead to normative statements such as “pretended meaning” and “economies of scale are a myth in many industries”.
        Overall, though the article provides evidence and examples supporting the claim that large firms are worse-off, its sources and claims are not credible and its arguments are rarely balanced, meaning that the conclusion remains unclear.
        """
        },


        {
            'role': 'user', 'content': f"""
        Extract: {extract}

        Question: {question}

        Mark Scheme: {markscheme}

        Remember to answer in continuous prose and follow the guideline for this question based on its number of marks, as provided in the system prompt.
        """
        }
        ],
                stream=False,
                temperature=0.3
            )
        print(response.choices[0].message.content)
        responses.append(response.choices[0].message.content)
        df = pd.DataFrame([])
        df['Extract'] = extracts[:i+1]
        df['Question'] = questions[:i+1]
        df['Mark Scheme'] = answers[:i+1]
        df['Response'] = responses 
        df.to_csv('al_econ_data_response_deepseek_r1_distill_v2.csv')
    except:
        pass