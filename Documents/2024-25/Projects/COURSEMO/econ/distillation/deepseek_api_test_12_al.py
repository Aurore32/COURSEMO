from openai import OpenAI
import pandas as pd

# sk-61dd5e4c06c4450893eb429ccf00e0fd: DeepSeek API key

client = OpenAI(api_key="sk-61dd5e4c06c4450893eb429ccf00e0fd", base_url='https://api.deepseek.com')
responses = []
actual_questions = []
actual_mark_schemes = []

df = pd.read_csv('./econ/structured/al_econ_structured_dataset_12_mark_al.csv')
questions = df['Question']
markschemes = df['Mark Scheme']
for i in range(len(questions)):
    question = questions[i]
    markscheme = markschemes[i]

    try:
        response = client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {"role": "system", "content": """
                        You are an A-Level economics teacher with full knowledge of the A-Level syllabus. You will be given an 12-mark **A-Level** (not AS-Level) structured question, as well as the mark scheme to the question, which should be referenced to generate a response that would obtain full marks according to it. Generate responses in continuous prose (no headings, bullets, or markdown). Follow these rules strictly:
        
        Formatting & Structure
        1. Never use bullets, lists, or headings—write in plain paragraphs.
        2. Stick to economic theory - do not reference real-world examples, even if given in the mark scheme. Your answer should be something a top-scoring student would write in an actual exam.
        3. Your essay should follow this structure:
        - The question will either be purely explanatory (it will ask you to "Explain" an economic statement in detail) or evaluative (it will ask you to "Consider" or "Discuss" a statement, i.e. give a balanced overview of why the statement may be true, why it may not be true, and an evaluation of whether it is true)
        - If the question is purely explanatory (no "Consider" or "Discuss" or "agree or disagree", mark scheme does not mention an "evaluation"), then:
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
                - Explain the point with a cause-and-effect chain if applicable ("Firms may choose to collude because of the Prisoner's Dilemma innate in an oligopoly. If a firm sets its price high, all other firms will set their prices low and the firm setting high prices will lose out on all its consumers; this results in a suboptimal Nash equilibrium and firms can counteract this by agreeing to set prices high together.")
        - If the question countains an evaluation part ("And consider whether...", "Discuss...", "Evaluate...", mark scheme mentioning an "evaluation"):
             - Follow all of the above for the explanation part.
             - When the mark scheme mentions a "discussion" or an "evaluation", you will need to write a paragraph that evaluates the statement in the question, i.e. consider several factors which may affect whether the statement is true.
             - The mark scheme may contain possible evaluation points. If not, think of your own (the time period in question, or the type of good, or the current state of the economy (recession or not), or the budget position of the governments, etc.)
             - You will only need to offer 3 to 4 evaluation points plus a conclusion that combines all the points. However, please elaborate on each of the points and explain them in depth with full chains of analysis, as above.
             - This should be at the very end of your essay.
             - Use conditional language ("This depends on...", "However, if...") for each point. 
             - Offer a conclusion combining all the evauation points.

        - The mark scheme may also mention level descriptors ("Level 4", "Level 4", etc.) The highest level (e.g. Level 4) will be the one for top marks. Please make sure you have followed the level descriptors for the highest level.

        Diagrams
        - If your answer needs to reference a diagram:
            - Indicate this via the text (DIAGRAM: (description of what the diagram is showing)).
            - Reference and explain key elements (e.g., "As shown in the monopoly diagram, profit maximization at MR=MC leads to a deadweight loss triangle ABC").
            - Link to analysis (e.g., "The shift from S1 to S2 in the supply diagram illustrates how subsidies reduce production costs").

        Non-Negotiable
        - DO NOT generate any headings, bold text or formatting. Generate your text in plain prose.
        - DO NOT include any real-world examples in your answer. Economic analysis only.          
        - Your explanation paragraphs combined should at least be two times longer than your evaluation, if the question asks for it. Expand on the analysis points from the mark scheme more than on evaluation.             
        - Your response will be approximately 800 words.
        """},
        {
            'role': 'user', 'content': f"""
        Question:     
        Explain what is meant by an oligopoly market and why prices might fluctuate less in an oligopoly market than in a perfectly competitive market. [12]

        
        Mark Scheme:
        4(b) 
Explain what is meant by an oligopoly market and why prices might fluctuate less in an oligopoly market than in a perfectly competitive market. [12]
An explanation of the characteristics of an oligopoly and of perfect competition should be provided. Oligopolies will be dominated by a few large firms which are interdependent. Perfect competition will have many small homogeneous firms which are not dependent on each other.
An explanation of price determination in an oligopoly. Possible points may include: price competition with the kinked demand curve. When one firm lowers prices, interdependence makes other firm lower prices; when one firm raises prices, all other firms will lower prices and the firm with raised prices will not be competitive. This leads to a kinked demand curve and price rigidity. Possible labeled diagram of the kinked demand curve.
The Prisoner's Dilemma and Nash equilibrium of all firms setting low prices; no firm has incentive to change their pricing decision.
An explanation of price leadership and link to price rigidity. Possible explanation of non-price competition and link to price rigidity. Conclusion to summarize price rigidity in an oligopoly.

L4 (9–12 marks): For a clear explanation of what is meant by an oligopoly and a sound analysis to link this to price rigidity through the kinked demand curve. Some reference to alternative non pricing market strategies might form part of a conclusion.
L3 For a competent comment but with limited development of the analysis and a weaker
conclusion. [7–8]
 L2 For a brief explanation of the characteristics and a weak comment on the price rigidity
and no conclusion [5–6]
 L1 For an answer which has some basic correct facts but includes irrelevancies. Errors of
theory or omissions of analysis will be substantial. [1–4]
        """
        },
        {
            'role': 'assistant', 'content': f"""

An oligopoly market refers to a market structure dominated by a few relatively large firms, usually producing many different brands of products. It is characterized primarily by interdependence between firms: each firm is sufficiently large so that decisions taken by it will affect decisions taken by all other firms as well. An oligopoly is also characterized by a high degree of market concentration, imperfect knowledge, and high levels of barriers to entry, resulting in each firm having some level of price-making powers; as such, each firm makes its pricing decisions based on how other firms will respond to such decisions, leading to there being no single theory of price determination in an oligopoly market. On the other hand, perfect competition describes an ideal market structure consisting of a large number of homogeneous small firms producing undifferentiated products which are perfect substitutes. The market is characterized by perfect information and nonexistent barriers to entry or exit; each firm has a very limited amount of market power, resulting in them becoming price takers. As such, firms have no power over their selling price, which is also their marginal revenue; they maximize profit by producing where their individual marginal costs equal the market-wide marginal revenue, which is determined by market forces of demand and supply rather than the decisions of an individual firm.

Oligopoly markets typically experience less price fluctuation than a perfectly competitive market because of how prices are determined in oligopoly markets. There are several ways in which an oligopoly market may determine its prices. First is price competition; firms out-compete other firms and attract customers by setting lower prices, but interdependence in an oligopoly means that if one firm sets lower prices, other firms will respond to that decision by setting even lower prices in turn. As such, the demand curve facing a firm when it tries to decrease prices from the market equilibrium can be characterized as very price inelastic (PED < 1): a reduction in price will not increase demand by much, as other firms will also reduce their prices in response and consumers will not change their decisions. If instead a firm decides to increase their prices in hopes of earning higher profits, all other firms will lower prices in response to attract more customers; as a result, the firm setting higher prices will become less desirable than all other firms in the market and demand will fall sharply. This can be characterized as a price-elastic (PED > 1) demand curve above the equilibrium price: a rise in price leads to a sharp fall in demand. All in all, price competition in an oligopoly is characterized by an inelastic demand curve below the market equilibrium and an elastic demand curve above it; this is known as a kinked demand curve, as pictured below:

(DIAGRAM: kinked demand curve.)

Either increasing or decreasing prices would sharply hurt a firm’s profits; remaining at the current market equilibrium price is the optimal decision for all firms, leading to price rigidity. This stands in contrast to a firm in a perfectly competitive market, whose pricing decisions are not determined by the firm itself, but rather by market forces of supply and demand that may fluctuate based on consumer tastes and other economic conditions. 

Other forms of price determination within an oligopoly also explain its price rigidity relative to perfectly competitive markets. An alternate theory of price competition concerns the Prisoner’s Dilemma inherent within an oligopoly market: as stated, if a firm sets its price high, all other firms will set their prices low and the firm setting high prices will lose out on all its consumers. As such, the optimal decision for all firms without knowing the decision of any other firm – a condition which holds in a market with imperfect information, such as an oligopoly – is to set low prices; this results in a Nash equilibrium where every firm is making their own optimal decision and has no incentive to change without changes being made by another firm. However, such an equilibrium is not the absolutely optimal outcome; if all firms set their prices high without undercutting any other firm, the level of profits present in the entire market would exceed the Nash equilibrium. This frequently gives rise to collusive behaviors between firms, either through a formal agreement to fix high prices and form a cartel – an entity which acts like a monopoly and sets its own prices, impervious to market forces of supply and demand, resulting in more price rigidity than perfect competition – or through tacit collusion and price leadership. Price leadership involves no formal agreements; instead, firms in an oligopoly follow the pricing decisions made by another firm which is seen as the dominant firm in the market, or the most knowledgeable firm. Firms have an incentive to follow another firm’s pricing decisions because they often want to avoid price wars arising from repeated price undercutting of other firms, which lead to unsustainably low prices that hurt every firm involved; as such, the decisions made by one firm will often set rigid prices for the rest of the market – this is known as barometric price leadership. 

As such, oligopolistic firms tend to avoid price competition; instead, firms either remain in a stable equilibrium marked by a kinked demand curve and are not incentivized to change prices, or follow the pricing decisions of a price leader and do not fluctuate their prices beyond that. Instead, oligopolistic firms compete through non-price competition such as reducing costs through research and development, producing differentiated products or a wider variety of products to satisfy consumers, or advertising their brand – none of which alter the rigidity of the market price.
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
        '''df = pd.DataFrame([])
        df['Extract'] = extracts[:i+1]
        df['Question'] = questions[:i+1]
        df['Mark Scheme'] = answers[:i+1]
        df['Response'] = responses 
        df.to_csv('al_econ_data_response_deepseek_r1_distill_v2.csv')'''
    except:
        pass

new_df = pd.DataFrame([])
new_df['Question'] = questions
new_df['Mark Scheme'] = markschemes
new_df['Response'] = responses
new_df.to_csv('./econ/distillation/al_econ_12_mark_a_level_deepseek_r1.csv')