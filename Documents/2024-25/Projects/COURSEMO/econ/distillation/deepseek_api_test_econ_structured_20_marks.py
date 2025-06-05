from openai import OpenAI
import pandas as pd

# sk-61dd5e4c06c4450893eb429ccf00e0fd: DeepSeek API key

client = OpenAI(api_key="sk-61dd5e4c06c4450893eb429ccf00e0fd", base_url='https://api.deepseek.com')
responses = []

#for i in range(len(questions)):
#    extract = extracts[i]
 #   markscheme = answers[i]
 #   question = questions[i]
df = pd.read_csv('./econ/structured/al_econ_structured_20_mark.csv')
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
                        You are an A-Level economics teacher with full knowledge of the A-Level syllabus. You will be given a 20-mark A-Level structured question, as well as the mark scheme to the question, which should be referenced to generate a response that would obtain full marks according to it. Generate responses in continuous prose (no headings, bullets, or markdown). Follow these rules strictly:
        
        Formatting & Structure
        1. Never use bullets, lists, or headings—write in plain paragraphs.
        2. Stick to economic theory - do not reference real-world examples, even if given in the mark scheme. Your answer should be something a top-scoring student would write in an actual exam.
        3. Your essay should follow the following two-part structure:

            Part 1: Analysis. Roughly 700 words.

            - The question will give you an economic statement to assess. This part will be your explanation for why the statement is justified by economic theory.
            - You should begin with a complete economics definition of all the key terms. The terms to define will be listed in the mark scheme.
            - Then, you should explain fully why the statement is true from a theoretical perspective. The content you should include to explain this will be given under AO1 and AO2 Knowledge and understanding and analysis in the mark scheme.
            - Your task will be to expand and elaborate on the points included within the mark scheme. The points in the mark scheme are directions only; treat them as short summaries of the points you should actually be writing in the essay.
            - For every point the mark scheme mentions, you will need to include:
                - Precise definitions of terms mentioned in the point, if necessary (e.g., “Allocative efficiency is…”).
                - Step-by-step, detailed chains of analysis (e.g., “Firms overproduce because MPC < MSC → this causes… which leads to…”).
                - Diagrams described textually (axes, curves, labels, welfare loss areas), if appropriate.
                - A hypothetical example explained fully, if necessary.
                - This will be a "mini-essay" that contains very detailed analysis, roughly 100-150 words long.
            - DO NOT use evaluative language in this section, like "however" or "this depends on". Speak positively about the question. DO NOT generate section titles, e.g. "Part 1" or "Part 2".
            - Example of mark scheme elaboration:
            Mark scheme: "Analysis of the links between collusion, higher prices and allocative efficiency should be provided."
            Response: "Traditional economic theory posits that collusion will lead to both higher prices and a less efficient allocation of resources because such collusion usually manifests in the form of a monopolistic cartel, which is less preferable to perfectly or monopolistically competitive firms in these regards. Both formal agreements and tacit understandings between firms to collude will typically involve each firm facing a production quota which restricts their own supply:

    (DIAGRAM: marginal cost, revenue and average total cost of individual firm, plus quota from collusion)

    Per agreement, the cartel decides on a price P*, which exceeds the price under a non-collusive oligopoly, for all firms to sell at. While individual firms would typically produce at the allocatively efficient intersection of marginal cost and marginal revenue (price), marked Q*, such a quantity produces too much supply for market price to settle at P*; as such, the cartel agreement enforces a strict quota upon each firm to produce at exactly Q1, which is not allocatively efficient. The combination of all firms’ quantities produced results in the diagram below:

    (DIAGRAM: monopoly diagram of cartel)

    In effect, the cartel has been able to make its own price which all firms in the oligopoly market accept, and thus acts as a single entity – a monopoly. This leads to a market equilibrium at price P* and quantity Q*, of which prices have been inflated and quantity contracted compared to a non-cartel oligopoly. As marked in gray, such an entity incurs a deadweight loss due to its quotas forcing all firms to produce at a quantity lower than what is allocatively efficient – MC  = MR = P. Per this analysis, cartels lead to both a higher price and a less efficient outcome compared to price competition, where firms are incentivized to collectively set low prices to not be out-competed by a cheaper product."
            

            Part 2: Evaluation. Roughly 250 words.
            
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
        - Your analysis paragraph should be much longer than your evaluation. Expand on the analysis points from the mark scheme more than on evaluation.             
        """},
        {
            'role': 'user', 'content': f"""
        Question:
        Some firms in oligopoly markets choose to collude rather than engage in price competition. This will lead to higher prices and a less efficient allocation of resources. Evaluate this statement. [20]

        Mark Scheme:
        Some firms in oligopoly markets choose to collude rather than engage
    in price competition. This will lead to higher prices and a less efficient
    allocation of resources.
    Evaluate this statement.
    Use Table A: AO1 Knowledge and understanding and AO2 Analysis and
    Table B: AO3 Evaluation to mark candidate responses to this question.
    AO1 and AO2 out of 14 marks. AO3 out of 6 marks.
    Indicative content
    Responses may include:
    AO1 Knowledge and understanding and AO2Analysis
    • The key characteristics of an oligopoly market should be described.
    • Collusion should be explained and compared to the alternative of price
    competition.
    • Some explanation of what is meant by an efficient allocation of
    resources.
    • Analysis of the links between collusion, higher prices and allocative
    efficiency should be provided.
    • Analysis of alternative price and non-price competitive approaches by
    oligopolistic firms should be discussed.
    • A large, dominant organization might have less incentive to operate
    efficiently and X inefficiencies might occur.
    AO3 Evaluation
    • Colluding oligopolistic firms might argue that an increase in size will
    lead to more opportunities to provide a wide range of goods and
    therefore more consumer choice.
    • It is possible that collusion will lead to an increase in dynamic efficiency
    and this increase would allow long term more investment and ultimately
    lower prices. A supporting diagram can be provided.
    • Such a decision will enable the firms to benefit from economies of
    scale. For example, reducing managerial inefficiencies. This could also
    lead to lower prices.
    • Any potential negative effects of collusion can be accounted for by
    government intervention and regulation after the collusion has taken
    place.
    • A conclusion should refer to the preceding analysis and then make
    some attempt to make a judgement whether, on balance, collusion will
    have a net negative effect on prices and/or efficiency.
    Accept all valid responses.
    AO1 Knowledge and understanding and AO2Analysis 14
    AO3 Evaluation 6
        """
        },
        {
            'role': 'assistant', 'content': f"""
    An oligopoly refers to a market structure dominated by a few large firms, with a five-firm concentration ratio of more than 50%; aside from its concentration ratio, oligopolies are characterized by high barriers to entry, a relatively high degree of price-setting ability from all dominant firms, and interdependence between firms in the sense that a decision taken by one firm will affect all others. 

    Collusion in oligopoly markets refers to the potentially unlawful collaboration between firms through agreements, meetings or insider trading to fix prices to the firms’ mutual advantage; it stands in opposition to price competition, which refers to firms strategically setting prices lower than the prices of other firms in an oligopoly to attract customers through the advantage of providing a cheaper product. Firms are motivated to collude due to the Prisoner’s Dilemma that occurs in an oligopoly market: without knowing the prices set by other firms, the optimal decision for all firms is to set prices low, resulting in a Nash equilibrium that marks the loss of potential profits compared to an outcome where all firms had set a higher and identical price. 

    Whether a market outcome leads to a less efficient allocation of resources can be primarily judged by two types of efficiency: allocative efficiency, which refers to producing at the quantity where the price of the product equals the marginal cost of the product such that every unit of the product produces imparts a marginal social benefit, and productive efficiency, which refers to producing at the lowest possible average total cost. 

    Traditional economic theory posits that collusion will lead to both higher prices and a less efficient allocation of resources because such collusion usually manifests in the form of a monopolistic cartel, which is less preferable to perfectly or monopolistically competitive firms in these regards. Both formal agreements and tacit understandings between firms to collude will typically involve each firm facing a production quota which restricts their own supply:

    (DIAGRAM: marginal cost, revenue and average total cost of individual firm, plus quota from collusion)

    Per agreement, the cartel decides on a price P*, which exceeds the price under a non-collusive oligopoly, for all firms to sell at. While individual firms would typically produce at the allocatively efficient intersection of marginal cost and marginal revenue (price), marked Q*, such a quantity produces too much supply for market price to settle at P*; as such, the cartel agreement enforces a strict quota upon each firm to produce at exactly Q1, which is not allocatively efficient. The combination of all firms’ quantities produced results in the diagram below:

    (DIAGRAM: monopoly diagram of cartel)

    In effect, the cartel has been able to make its own price which all firms in the oligopoly market accept, and thus acts as a single entity – a monopoly. This leads to a market equilibrium at price P* and quantity Q*, of which prices have been inflated and quantity contracted compared to a non-cartel oligopoly. As marked in gray, such an entity incurs a deadweight loss due to its quotas forcing all firms to produce at a quantity lower than what is allocatively efficient – MC  = MR = P; because the cartel, acting as a monopoly, now faces no competition, it has no incentive to minimize its average total costs and will suffer X-inefficiency as well as productive inefficiency for not producing at the lowest point on its ATC curve. Per this analysis, cartels lead to both a higher price and a less efficient outcome compared to price competition, where firms are incentivized to collectively set low prices to not be out-competed by a cheaper product. The outcome is also less efficient compared to non-price competition, where oligopoly firms rush to reduce their costs, innovate, and produce differentiated products for consumers, which is likely to increase efficiency via an increased diversity of choice.

    However, it could be argued that certain efficiencies are to be gained from collusion; in particular, the increase in scale that firms benefit from by operating as a colluding cartel, versus operating as smaller individual firms, can lead to increased efficiency in several ways. If the colluding firms in the oligopoly collaborate not only in price fixing, but also in day-to-day operations – for instance, merging their spending on marketing, innovation, or research and development – then there is dynamic efficiency to be gained: a firm of a larger scale has more to spend on R&D for new and greater varieties of products, potentially leading to more long-term investment, reductions in costs, and, ultimately, lower prices and higher levels of consumer choice. If X-inefficiency does not set in, colluding firms will also benefit from economies of scale: for instance, shared managerial staff or shared departments across the two firms may lead to reduced labor costs, and purchasing inputs in bulk will likely result in lower production costs across the board, potentially leading to higher levels of productive efficiency which may manifest as lower prices for consumers in the long run. The negative impacts of collusion – higher prices for consumers in the short-term, increased barriers of entry due to monopolization – can also be mitigated by regulatory policies such as fines or antitrust policies, which would reduce the harmful impacts of cartels on consumers while preserving the dynamic efficiency they bring. Overall, economic theory posits that collusion will lead to higher prices and less efficient market outcomes in the short-term; however, the long-term effects are uncertain and may even be positive, dependent on whether the colluding firms are willing to collaborate on research and development, whether such collaboration results in economies of scale, and whether such economies of scale trickle down to reduced prices in consumers’ hands.
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
new_df.to_csv('al_econ_20_mark_deepseek_r1.csv')