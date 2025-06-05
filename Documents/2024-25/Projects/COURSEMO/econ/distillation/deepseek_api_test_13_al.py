from openai import OpenAI
import pandas as pd

# sk-61dd5e4c06c4450893eb429ccf00e0fd: DeepSeek API key

client = OpenAI(api_key="sk-61dd5e4c06c4450893eb429ccf00e0fd", base_url='https://api.deepseek.com')
responses = []
actual_questions = []
actual_mark_schemes = []

df = pd.read_csv('./econ/structured/al_econ_structured_dataset_13_mark.csv')
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
                        You are an A-Level economics teacher with full knowledge of the A-Level syllabus. You will be given a 13-mark **A-Level** (not AS-Level) structured question, as well as the mark scheme to the question, which should be referenced to generate a response that would obtain full marks according to it. Generate responses in continuous prose (no headings, bullets, or markdown). Follow these rules strictly:
        
        Formatting & Structure
        1. Never use bullets, lists, or headings—write in plain paragraphs.
        2. Stick to economic theory - do not reference real-world examples, even if given in the mark scheme. Your answer should be something a top-scoring student would write in an actual exam.
        3. Please start the essay directly, without an introduction. Move on to definitions immediately.
        4. Your essay should follow this structure:
        - The question will usually ask you to "Consider" or "Discuss" an economic statement statement, or ask you whether you "agree or disagree".
        - This requires an answer in two parts. Aim to reach roughly 800 words total.
        - The first part of your answer should be an economic analysis and explanation of the statement you are considering. 
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
        - The second part of your essay will be an evaluation part, if the question countains an evaluation part ("And consider whether...", "Discuss...", "Evaluate...", mark scheme mentioning an "evaluation"):
             - If the question does not have an evaluation part, simply conclude the explanation part above.
             - When the mark scheme mentions a "discussion" or an "evaluation", you will need to write a paragraph that evaluates the statement in the question, i.e. consider several factors which may affect whether the statement is true.
             - Even if the mark scheme deos not mention a "discussion" or an "evaluation, if the question uses the word "discuss" or "consider" or "assess" or "agree or disagree", you will need to write an evaluation.
             - The mark scheme may contain possible evaluation points. If not, think of your own (the time period in question, or the type of good, or the current state of the economy (recession or not), or the budget position of the governments, etc.)
             - This should be at the very end of your essay.
             - Use conditional language ("This depends on...", "However, if...") for each point. 
             - Offer a conclusion combining all the evauation points.

        - The mark scheme may also mention level descriptors ("Level 4", "L4", etc.) The highest level (e.g. Level 4) will be the one for top marks. Please make sure you have followed the level descriptors for the highest level.

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
        Discuss whether a firm in monopolistic competition is more likely to be in the public interest than a firm that is a monopoly. [13]

        
        Mark Scheme:
        (b) Discuss whether a firm in monopolistic competition is more likely to be in the public
interest than a firm that is a monopoly. [13]

Discussion of the characteristics of the two types of market structure. Long run profit will be
normal in monopolistic competition but not necessarily so in monopoly. Output is likely to be
lower in monopoly; prices are likely to be higher. However, there may be merits of large
scale monopoly firms through economies of scale. It may be difficult to make a general
overall judgement.
 L4 For a sound discussion with good explanation of the analysis and a clear understanding
of the differences between the two market structures together with a conclusion [9–13]
 L3 For a competent comment but with more limited discussion of the two market structures
but still with a conclusion [7–8]
 L2 For a brief explanation of the structures, limited discussion and no conclusion [5–6]
 L1 For an answer which has some basic correct facts but includes irrelevancies. Errors of
theory or omissions of analysis will be substantial. [1–4] 
        """
        },
        {
            'role': 'assistant', 'content': f"""
According to economic theory a monopoly exists when there is one firm in the industry. It is characterized by selling a unique product and high entry barriers. Compared with other market structures, demand under monopoly will be relatively inelastic at each price. This gives monopolist substantial powers to charge a high price. Nevertheless, it is still constrained by its demand curve i.e. a rise in price will lower the quantity demanded. In order for a firm to maintain its monopoly position there must be barriers to the entry of new firms. Barriers can be of various forms, such as economies of scale, high capital cost, legal barriers and various marketing barriers.

Monopolistic competition incorporates features of both perfect competition and monopoly. As with perfect competition, there is freedom of entry into the industry; there are many relatively small firms in the market, each producing goods which are slightly different from their competitors. The existence of product differentiation means that firms have a certain degree of monopoly power; thus, if they raise their price they do not lose all their buyers even though they produce products which are close substitutes. The result is a downward sloping relatively elastic demand curve. As a result, a monopolistically competitive firm is not a price taker. Product differentiation can be reinforced through advertising which produces an element of brand loyalty.

Under monopolistic competition, freedom of entry eliminates supernormal profit and forces the firm to produce where its AR = AC; in the long-run, prices are kept down as any supernormal profits will lead to new firms joining the market, reducing demand for any individual firm’s products and thereby reducing prices, forcing every firm to operate at a level of normal profit which minimizes average total costs. Under monopoly, however, barriers to entry allow profits to remain supernormal in the long run; no profits are “competed away” by other firms joining the market. Hence the monopolist is not forced to produce where its AR = AC. Thus, other things remaining the same, a comparison of profit maximizing position for an industry under monopoly with that of a firm in monopolistic competition reveals that long run price will tend to be higher and output lower under monopoly. 

Whether a monopolistically competitive firm is more in the public interest than a monopolistic firm can be assessed through the lens of several types of efficiencies. Neither type of firm is allocatively efficient, characterized by producing where price equals marginal cost and every unit produced confers a net marginal benefit to society, thus reaching an optimum in terms of maximizing social welfare; both types of firms face a downward-sloping demand curve and thus produce at a point where price exceeds marginal cost. In this sense, neither type of firm is in the public interest.

In terms of productive efficiency, which refers to producing at the minimum possible average total cost (ATC), a monopolistically competitive market will produce at exactly the quantity where ATC is minimized in the long-run; 
competition and availability of close substitutes requires the firm in a monopolistically competitive market to use the most efficient known technique to retain normal profits.  In contrast, the monopolist sheltered by barriers to entry can still make large profits even if it is not using the most efficient technique, thus having no incentive to produce at the productively efficient minimum ATC or to reduce its costs over time; this is known as X-inefficiency, and also results in productive inefficiency.

In addition to this monopolies may lack the incentive to introduce new product varieties and large monopolies may be able to exert political pressure and thereby get favourable treatment from the government. It could be argued that public interest would be better served through the degree of product variety and consumer choice present with a monopolistically competitive firm, which offers a differentiated product different from the rest of the market, unlike a monopoly.

However, this analysis depends on several economic factors which determine the efficiencies of the two types of firms. First, monopolies operate on a larger scale than monopolistically competitive firms, which are assumed to be small with little market power; as such, monopolies could benefit from several forms of economies of scale, for instance bulk-buying economies of scale where inputs are sourced at a cheaper rate in bulk or managerial economies of scale from centralized administration. In the long-run, the economies of scale enjoyed by a monopoly would lead to lower average total costs which, despite not being productively efficient, could be less costly than that of a monopolistically competitive firm and result in lower prices being charged for the product, which is in the public interest as far as consumers are concerned. Furthermore, monopolies retain some measure of supernormal profit as a result of barriers to entry, while monopolistically competitive firms do not; a monopolist could re-invest excess profits into research and development or innovation, leading to improvements in product quality and a dynamic efficiency marked by a reduction in costs over time – an efficiency which monopolistically competitive firms will not be able to achieve. Finally, government regulation may be able to curb the societally harmful aspects of a monopoly while retaining its benefits to the public interest: for instance, a quota on the monopolist to produce at the exact quantity where price equals marginal cost will force the monopolist to become allocatively efficient, assuming that such regulatory policies do not require too much excess costs on the government  to enforce. 

In conclusion, although both monopolies and monopolistically competitive firms are allocatively inefficient, monopolistically competitive firms generally serve the public interest more by producing a greater variety of goods and achieving productive efficiency by minimizing average total costs; however, in cases where monopolies can benefit from economies of scale and re-invest excess profits into research and development, and where monopolies are carefully regulated by the government, the dynamic efficiency provided by a monopoly may serve the public interest more in the long-run by reducing costs and increasing product qualities.
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
        new_df.to_csv('./econ/distillation/al_econ_13_mark_deepseek_r1.csv')
        '''df = pd.DataFrame([])
        df['Extract'] = extracts[:i+1]
        df['Question'] = questions[:i+1]
        df['Mark Scheme'] = answers[:i+1]
        df['Response'] = responses 
        df.to_csv('al_econ_data_response_deepseek_r1_distill_v2.csv')'''
    except:
        pass
