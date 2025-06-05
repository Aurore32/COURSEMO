question = ''

context = '''
Essay question: {}

Syllabus topics (separated by lines):

--- AS Level ---

1.1 Scarcity, choice and opportunity cost 
1.2 Economic methodology 
1.3 Factors of production
1.4 Resource allocation in different economic systems
1.5 Production possibility curves
1.6 Classification of goods and services
2.1 Demand and supply curves
2.2 Price elasticity, income elasticity and cross elasticity of demand
2.3 Price elasticity of supply
2.4 The interaction of demand and supply
2.5 Consumer and producer surplus
3.1 Reasons for government intervention in markets
3.2 Methods and effects of government intervention in markets
3.3 Addressing income and wealth inequality
4.1 National income statistics
4.2 Introduction to the circular flow of income
4.3 Aggregate Demand and Aggregate Supply analysis
4.4 Economic growth
4.5 Unemployment
4.6 Price stability
5.1 Government macroeconomic policy objectives
5.2 Fiscal policy
5.3 Monetary policy
5.4 Supply-side policy
6.1 The reasons for international trade
6.2 Protectionism
6.3 Current account of the balance of payments
6.4 Exchange rates
6.5 Policies to correct imbalances in the current account of the balance of payments

--- A-Level ---

7.1 Utility
7.2 Indifference curves and budget lines
7.3 Efficiency and market failure
7.4 Private costs and benefits, externalities and social costs and benefits 
7.5 Types of cost, revenue and profit, short-run and long-run production
7.6 Different market structures
7.7 Growth and survival of firms
7.8 Differing objectives and policies of firms
8.1 Government policies to achieve efficient resource allocation and correct market failure
8.2 Equity and redistribution of income and wealth
8.3 Labour market forces and government intervention
9.1 The circular flow of income
9.2 Economic growth and sustainability
9.3 Employment/unemployment
9.4 Money and banking
10.1 Government macroeconomic policy objectives
10.2 Links between macroeconomic problems and their interrelatedness
10.3 Effectiveness of policy options to meet all macroeconomic objectives
11.1 Policies to correct disequilibrium in the balance of payments
11.2 Exchange rates
11.3 Economic development
11.4 Characteristics of countries at different levels of development
11.5 Relationship between countries at different levels of development
11.6 Globalisation

'''.format(question)

chat = [{
            "content": f"You are an A-Level economics teacher with full knowledge of the A-Level economics syllabus, trying to help students practice their essays. You will be given an A-Level economics essay question, as well as a list of AS-Level (11th grade) and A-Level (12th grade) economics topics on the syllabus; you will be told which topics are AS-level and which are A-Level. Your task is to directly identify which of the given topics are most related to the given question; if more than one topic is closely related to the question, please state all of them. You should state the topic directly (or, for more than one topic, separated with commas), without additional explanation. Additionally, please also state directly in a new line whether you think the question is an A-Level question or an AS-Level question, based on the topics given. \n",
            "role": "system"
        },
        {
            "content": context,
            "role": "user"
        }]

FastLanguageModel.for_inference(model) # Enable native 2x faster inference
inputs = tokenizer.apply_chat_template(
    chat,
    tokenize = True,
    add_generation_prompt = True, # Must add for generation
    return_tensors = "pt",
).to("cuda")

outputs = model.generate(input_ids = inputs, max_new_tokens = 1024, use_cache = True)
response = tokenizer.batch_decode(outputs)[0]
print(response)







"You are an A-Level Physics teacher with full knowledge of the A-Level Physics syllabus, trying to help students craft a sample test paper full of "




















question = ''
answer = '''

'''
mark = ''
context = '''
Essay question: {}

Mark scheme: {}

Number of marks: {}
'''.format(question, answer, mark)

chat = [{
            "content": f"You are an A-Level economics teacher with full knowledge of the A-Level economics syllabus. Your task is to write a sample essay that will obtain top marks according to the A-Level mark scheme on the given essay question. You will be provided with the mark scheme to the question. For some questions, the mark scheme will detail a three-part structure based on the criteria AO1 (Knowledge), AO2 (Analysis), and AO3 (Evaluation); your task is to combine the requirements of the mark scheme into an A-Level economics essay that is coherent and analytical. You will also be provided with the number of marks to the question. The length of the essay you write should roughly be in line with the number of marks; for instance, a 20-mark essay should be roughly 800 words and a 4-mark essay should be 150.\n",
            "role": "system"
        },
        {
            "content": context,
            "role": "user"
        }]

FastLanguageModel.for_inference(model) # Enable native 2x faster inference
inputs = tokenizer.apply_chat_template(
    chat,
    tokenize = True,
    add_generation_prompt = True, # Must add for generation
    return_tensors = "pt",
).to("cuda")

outputs = model.generate(input_ids = inputs, max_new_tokens = 1024, use_cache = True)
response = tokenizer.batch_decode(outputs)[0]
print(response)


"""You are an A-Level Physics teacher with full knowledge of the A-Level Physics syllabus, trying to create a set of practice problems for students to learn from. 
You will be given the entire list of concepts in the A-Level Physics syllabus, organized into broad categories (e.g. "2. Kinematics") and sub-categories under these broad categories (e.g. " (AS) 2.1 Equations of motion".) 
The sub-categories will be divided into AS-Level (11th grade), indicated with '(AS)', and A-Level (12th grade), indicated with '(A2)'.
You will be given a long-answer A-Level physics problem; your task is to classify the problem into a broad category and as many sub-categories as you believe is relevant. The sub-categories do not have to be under the broad category.
"""