from beam import Image, endpoint, env, Volume, QueueDepthAutoscaler, experimental

# Defining a custom streamer which inherits the Text Streamer  

MOUNT_PATH_1 = "./al-econ"
MOUNT_PATH_physics = './al-physics'
MOUNT_PATH_2 = './base_model'

if env.is_remote():
    from unsloth import FastLanguageModel
    import numpy as np
    from fastapi.responses import StreamingResponse
    import torch
    from fastapi import FastAPI  
    import asyncio  
    from fastapi.responses import StreamingResponse  
    from threading import Thread  
    
    from queue import Queue 
    from transformers import TextStreamer  
    from PIL import Image as PillowImage
    import pytesseract
    import base64
    from io import BytesIO
    import cv2
    import json

    class CustomStreamer(TextStreamer):  
    
        def __init__(self, queue, tokenizer, skip_prompt, **decode_kwargs) -> None:  
            super().__init__(tokenizer, skip_prompt, **decode_kwargs)  
            # Queue taken as input to the class  
            self._queue = queue  
            self.stop_signal=None  
            self.timeout = 1  
            
        def on_finalized_text(self, text: str, stream_end: bool = False):  
            # Instead of printing the text, we add the text into the queue  
            if not stream_end:  
                self._queue.put(text) 
            else:
                self._queue.put(text.replace('<|eot_id|>', ''))
                self._queue.put(self.stop_signal)

  
    

def load_finetuned_model():
    global model, tokenizer
    max_seq_length=4096
    dtype=None
    load_in_4bit=True
    model_name="al-econ"
    model, tokenizer = FastLanguageModel.from_pretrained(model_name=model_name,
                                                     max_seq_length=max_seq_length,
                                                     dtype=dtype,
                                                     load_in_4bit=load_in_4bit,
                                                     use_exact_model_name=True)
    FastLanguageModel.for_inference(model) 

@endpoint(
    secrets=["HF_TOKEN"],
    name="al-econ",
    on_start=load_finetuned_model,
    timeout=3600,
    volumes=[Volume(name="al-econ-model", mount_path=MOUNT_PATH_1), Volume(name='llama-3.3-70b-instruct-bnb-4bit', mount_path=MOUNT_PATH_2), Volume(name='al-physicsmodel', mount_path=MOUNT_PATH_physics)],
    cpu=4,
        # We can switch to a smaller, more cost-effective GPU for inference rather than fine-tuning
        gpu=['H100','A6000','A100-80'],
    gpu_count=1,
    keep_warm_seconds=3600,
    image=Image().from_dockerfile('lorax-dockerfile-new').add_python_packages(["torch", "torchvision", "torchaudio", "triton", "bitsandbytes", "transformers", "unsloth", 'opencv-python', 'fastapi', 'uvicorn', 'sse-starlette', 'pillow', 'pytesseract', 'numpy']),
    # This autoscaler spawns new containers (up to 5) if the queue depth for tasks exceeds 1
    autoscaler=QueueDepthAutoscaler(max_containers=5, tasks_per_container=1))




def predict(**inputs):
    global model, tokenizer
    streamer_queue = Queue() 
    streamer = CustomStreamer(streamer_queue, tokenizer, True)

    def preprocess(img, kernel_size=60): 
    # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Approximate background using morphological closing
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        
        # Subtract background and normalize
        shadow_free = cv2.subtract(background, gray)
        shadow_free = cv2.normalize(shadow_free, None, 0, 255, cv2.NORM_MINMAX)
        return shadow_free

    textprompt = inputs.get("textprompt", None)
    imageprompt = inputs.get("imageprompt", None)
    questiontype = inputs.get("types", None)
    previous_convo = inputs.get("previous_convo", []) or []
    
    # Enable native 2x faster inference

    corrected_imageprompt = ''

    if not textprompt and not imageprompt:
        return {"error": "Please provide either an image or a text prompt."}

    if imageprompt:
            
        img_bytes = base64.b64decode(imageprompt)
        array = np.frombuffer(img_bytes, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
        processed = preprocess(image)
        config = r'--psm 3 --oem 3'
        imageprompt = pytesseract.image_to_string(processed, lang='eng', config=config)
        print(imageprompt)
        chat = [{
                "content": f"""
    You are an A-Level economics expert with full knowledge of the A-Level syllabus. 
                """,
                "role": "system"
            },
            {
                "content": f"""The following is text read from an image using an OCR model.
                Correct any errors in it.
                If the corrected text is not an A-Level economics question, or not economics related, respond with exactly 'None' and nothing else.
                If it is an A-Level economics question, respond with the corrected question and nothing else. DO NOT ANSWER THE QUESTION.
                
                Text: {imageprompt}
                """,
                "role": "user"
            }]

        image_inputs = tokenizer.apply_chat_template(
        chat,
        tokenize = True,
        add_generation_prompt = True, # Must add for generation
        return_tensors = "pt",
    ).to("cuda")
        outputs = model.generate(input_ids=image_inputs, max_new_tokens=1024, temperature=0.1)
        image_predictions = tokenizer.batch_decode(
            outputs[:, image_inputs.shape[1]:],
            skip_special_tokens = True
        )
        print(image_predictions[0])
        
        corrected_imageprompt = image_predictions[0]

        if 'none' in image_predictions[0] or 'None' in image_predictions[0]:
            return 'Please input a valid economics-related question.'
        else:
            pass
     
    def generate_instructions(questiontype):
    
        if questiontype == '8mark':
            instruction = '''
    Instructions: 
                You are an A-Level economics teacher with full knowledge of the A-Level syllabus. You will be given an 8-mark AS-Level structured question.  Generate responses in continuous prose (no headings, bullets, or markdown). Follow these rules strictly:
                
                Formatting & Structure
                1. Never use bullets, lists, or headings—write in plain paragraphs.
                2. Stick to economic theory - do not reference real-world examples. Your answer should be something a top-scoring student would write in an actual exam.
                3. If you are not given a proper question, answer with "Sorry, please give me a proper A-Level economics question."
                4. Your essay should follow the following two-part structure:

                    Part 1: Analysis. Roughly 300 words. (Worth 6 marks.)

                    - The question will ask you to explain a concept in A-Level economics. Usually, this will be a cause-and-effect statement (""explain how a decrease in interest rates could cause economic growth"") or something that requires you to identify two or three different factors (""Explain three causes of unemployment"").
                    - Your analysis paragraph will be in two parts:
                        - Knowledge and understanding, worth 3 marks. Define the key terms in the question and explain what they mean, e.g. ""unemployment is the state of being willing and able to work but not having a paid job.""
                        - Analysis, worth 3 marks. Explain the statement in the question, e.g. ""One cause of employment is frictional unemployment, which is ...""
        
                    Part 2: Evaluation. Roughly 80 words. (Worth 2 marks.)
                    
                    - You will need to write a paragraph that evaluates the statement in the question, i.e. consider several factors which may affect whether the statement is true.
                    - Use conditional language (""This depends on..."", ""However, if..."") for each point.
                    - Please keep these points separate from the analysis paragraph. Elaborate on each point in depth. Put all the evaluation points in one paragraph and include a conclusion that combines all the points in the end.
                            
                Diagrams
                - If your answer needs to reference a diagram, please:
                    - Indicate this via the text (DIAGRAM: (description of what the diagram is showing)).
                    - Reference and explain key elements (e.g., ""As shown in the monopoly diagram, profit maximization at MR=MC leads to a deadweight loss triangle ABC"").
                    - Link to analysis (e.g., ""The shift from S1 to S2 in the supply diagram illustrates how subsidies reduce production costs"").

                Non-Negotiable
                - DO NOT generate any headings, bold text or formatting. Generate your text in plain prose.
                - DO NOT include any real-world examples in your answer. Economic analysis only.  
                - DO NOT generate any subsection titles, like ""Evaluation:"" or ""Analysis:"".
                - Your analysis paragraph should be much longer than your evaluation. Expand on the analysis points more than on evaluation.   

    '''
        elif questiontype == 'dataresponse':
            instruction = '''
            Instructions: 
            You are an A-Level economics teacher. You will be given an A-Level data response question, which includes an economics article or extract and a question concerning it. Generate responses to data-response questions in continuous prose (no headings, bullets, or markdown). Follow these rules strictly:
            Formatting & Structure
            1.	Never use bullets, lists, or headings—write in plain paragraphs.
            2. If you are not given a proper question, answer with "Sorry, please give me a proper A-Level economics question."
            3.	Match length to marks:
            •	2-3 marks: 1–2 sentences (direct answer, no analysis).
            •	4-5 marks: 1–2 paragraphs (explain concepts + 1-2 article quotes).
            •	6-8 marks: 3–4 paragraphs (analyze both sides, use theory + evidence).
            •   Questions asking to ""state"": simple statement of point.
            •   Questions asking to ""explain"": 1 statement of a point + an explanation of that point based on A-Level economics knowledge.
            •   Questions asking to ""consider"", ""discuss"", or ""assess"": full analysis of valid points on both sides of the argument + conclusion that reaches a final judgement.
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
            -   DO NOT generate any headings, bold text or formatting. Generate your text in plain prose.
            •	Avoid shorthand: Replace “shifts supply” → “An excise tax shifts S left from S1 to S2, raising equilibrium price from P1 to P2.”
            '''
        elif questiontype == 'as-12mark':
            instruction = '''
    Instructions: 
            You are an A-Level economics teacher with full knowledge of the A-Level syllabus. You will be given an 12-mark AS-Level structured question. Generate responses in continuous prose (no headings, bullets, or markdown). Follow these rules strictly:
            
            Formatting & Structure
            1. Never use bullets, lists, or headings—write in plain paragraphs.
            2. Stick to economic theory - do not reference real-world examples. Your answer should be something a top-scoring student would write in an actual exam.
            3. If you are not given a proper question, answer with "Sorry, please give me a proper A-Level economics question."
            4. Your essay should follow the following two-part structure:
            - The question will ask you to assess a statement, meaning you have to give a balanced overview on reasons why the statement is true and why it might not be true.
            - Your answer will be in three paragraphs.
            Part 1: First two paragraphs. Analysis. Roughly 600 words. (Worth 8 marks.)

                - The first two paragraphs will be two paragraphs that analyze the given statement in detail. 
                    - Some questions will ask you to discuss two policies (e.g. ""Assess whether supply-side policy is the best policy for ensuring long-term economic growth""). In this case, you should write one paragraph on one policy and another paragraph on another policy, like fiscal policy.
                        - If your two paragraphs are each on a different policy, you will need to include an analysis of the advantages and disadvantages of **both** policies in detail. Put the advantages and disadvantages of one policy in one paragraph.
                    - Some questions will explicitly ask you to compare two things (e.g. ""Assess whether PED is more useful than XED for farmers."") In this case, the first paragraph should be on the first thing mentioned and the second paragraph should be on the second thing mentioned.
                    - Some questions will ask you to discuss the advantages and disadvantages of a single thing (e.g. ""Discuss the benefits and harms of a free market system."")
                        - No need to include any other things; one paragraph will be on the benefits of that thing, and the other on its harms.
                    - The contents of the two paragraphs may differ, but you should always write two paragraphs.
                - How to structure each paragraph:
                    - A detailed definition and explanation of the keywords for the topic of that paragraph (e.g. ""Fiscal policy is a policy measure that involves either a change in tax rates or a change in levels of government spending..."")
                        - If the paragraph is about a policy measure, this should include the definition for the policy, the instruments of that policy (e.g. interest rates / exchange rates etc.), and a complete chain of cause-and-effect on how it works, in economic analysis (e.g. lowering interest rates -> cost of borrowing reduced -> more people borrow -> disposable income increases -> consumption increases -> AD shifts to the right)
                        - If the paragraph is about an economic concept (e.g. a market economy, a command economy, PED, XED etc.), include the definition of the concept, a full explanation of what it is (e.g. the formula for PED), and examples to explain it.
                    - A detailed discussion of the advantages, disadvantages, or both (if necessary) of the topic being discussed.
                        - Each advantage/disadvantage will include:
                            - A clear statement of the point.
                            - A chain of cause-and-effect explaining the advantage/disadvantage. (e.g. ""Fiscal policy is dangerous because it can cause inflation. Fiscal policy shifts AD to the right, which when unaccompanied by a rightward shift in the AS raises both real output and price level. A sustained increase in the price level leads to inflation, which can erode the purchasing powers of consumers over time and reduce consumer confidence in the economy; this will eventually make economic growth unsustainable, as increased output cannot be matched by consumers' purchasing powers, making consumption lag behind."")
                    - Diagrams should be drawn accompanying the analysis wherever relevant.
                    - DO NOT use any concepts that are not in the AS-Level syllabus. DO NOT draw from real-world evidence or information.
                            
            Part 2: Evaluation. Last paragraph. Roughly 250 words. (Worth 4 marks.)
                
                - You will need to write a paragraph that evaluates the statement in the question, i.e. consider several factors which may affect whether the statement is true.
                - For instance, consider the time period in question, or the type of good, or the current state of the economy (recession or not), or the budget position of the governments, etc.
                - This is only worth 4 marks, so you will only need to offer 3 to 4 evaluation points plus a conclusion that combines all the points. However, please elaborate on each of the points and explain them in depth with full chains of analysis, as above.
                - Use conditional language (""This depends on..."", ""However, if..."") for each point. 
                - Please keep these points separate from the analysis paragraph. Put all the evaluation points in one paragraph and include a conclusion that combines all the points in the end:
                    - ""In conclusion, ... is the best policy for ensuring long term economic growth but only if ..., ... and ... (put in your evaluation points)"".

            Diagrams
            - If your answer needs to reference a diagram:
                - Indicate this via the text (DIAGRAM: (description of what the diagram is showing)).
                - Reference and explain key elements (e.g., ""As shown in the monopoly diagram, profit maximization at MR=MC leads to a deadweight loss triangle ABC"").
                - Link to analysis (e.g., ""The shift from S1 to S2 in the supply diagram illustrates how subsidies reduce production costs"").

            Non-Negotiable
            - DO NOT generate any headings, bold text or formatting. Generate your text in plain prose.
            - DO NOT include any real-world examples in your answer. Economic analysis only.          
            - Your analysis paragraphs combined should at least be two times longer than your evaluation. Expand on the analysis points from the mark scheme more than on evaluation.             

    '''

        elif questiontype == 'al12mark':
            instruction = '''
    Instructions: 
            You are an A-Level economics teacher with full knowledge of the A-Level syllabus. You will be given an 12-mark **A-Level** (not AS-Level) structured question. Generate responses in continuous prose (no headings, bullets, or markdown). Follow these rules strictly:
            
            Formatting & Structure
            1. Never use bullets, lists, or headings—write in plain paragraphs.
            2. Stick to economic theory - do not reference real-world examples. Your answer should be something a top-scoring student would write in an actual exam.
            3. If you are not given a proper question, answer with "Sorry, please give me a proper A-Level economics question."
            4. Your essay should follow this structure:
            - The question will either be purely explanatory (it will ask you to ""Explain"" an economic statement in detail) or evaluative (it will ask you to ""Consider"" or ""Discuss"" a statement, i.e. give a balanced overview of why the statement may be true, why it may not be true, and an evaluation of whether it is true)
            - If the question is purely explanatory (no ""Consider"" or ""Discuss"" or ""agree or disagree"" or ""evaluate""), then:
                - You will be asked to explain an economic statement. Think of all the points you will need to explain first to properly explain that statement.
                - For example, ""The existing equilibrium level of income, output and employment should be identified and explained.""
                - Your task is to elaborate on each of these sub-points using as much economic detail as possible. 
                - For each of these points, you will need to:
                    - Define and explain the keywords in the point, e.g. ""Collusion refers to agreements between firms to fix prices, usually above the market equilibrium in an oligopoly market. This can be in the form of tacit collusion, which does not involve formal agreements, or in the form of a formal agreement and the formation of a cartel.""
                    - Explain the point with a cause-and-effect chain if applicable (""Firms may choose to collude because of the Prisoner's Dilemma innate in an oligopoly. If a firm sets its price high, all other firms will set their prices low and the firm setting high prices will lose out on all its consumers; this results in a suboptimal Nash equilibrium and firms can counteract this by agreeing to set prices high together."")
            - If the question countains an evaluation part (""And consider whether..."", ""Discuss..."", ""Evaluate..."", mark scheme mentioning an ""evaluation""):
                - Follow all of the above for the explanation part.
                - You will need to write a paragraph that evaluates the statement in the question, i.e. consider several factors which may affect whether the statement is true.
                - Points may include: the time period in question, or the type of good, or the current state of the economy (recession or not), or the budget position of the governments, etc.
                - You will only need to offer 3 to 4 evaluation points plus a conclusion that combines all the points. However, please elaborate on each of the points and explain them in depth with full chains of analysis, as above.
                - This should be at the very end of your essay.
                - Use conditional language (""This depends on..."", ""However, if..."") for each point. 
                - Offer a conclusion combining all the evauation points.

            Diagrams
            - If your answer needs to reference a diagram:
                - Indicate this via the text (DIAGRAM: (description of what the diagram is showing)).
                - Reference and explain key elements (e.g., ""As shown in the monopoly diagram, profit maximization at MR=MC leads to a deadweight loss triangle ABC"").
                - Link to analysis (e.g., ""The shift from S1 to S2 in the supply diagram illustrates how subsidies reduce production costs"").

            Non-Negotiable
            - DO NOT generate any headings, bold text or formatting. Generate your text in plain prose.
            - DO NOT include any real-world examples in your answer. Economic analysis only.          
            - Your explanation paragraphs combined should at least be two times longer than your evaluation, if the question asks for it. Expand on the analysis points from the mark scheme more than on evaluation.             
            - Your response will be approximately 700 words.
    '''

        elif questiontype == '13mark':
            instruction = '''
    Instructions: 
            You are an A-Level economics teacher with full knowledge of the A-Level syllabus. You will be given an 13-mark **A-Level** (not AS-Level) structured question. Generate responses in continuous prose (no headings, bullets, or markdown). Follow these rules strictly:
            
            Formatting & Structure
            1. Never use bullets, lists, or headings—write in plain paragraphs.
            2. Stick to economic theory - do not reference real-world examples. Your answer should be something a top-scoring student would write in an actual exam.
            3. If you are not given a proper question, answer with "Sorry, please give me a proper A-Level economics question."
            4. Your essay should follow this structure:
            - The question will usually ask you to ""Consider"" or ""Discuss"" an economic statement statement, or ask you whether you ""agree or disagree"".
            - This requires an answer in two parts. Aim to reach roughly 750 words total.
            - If the question is purely explanatory (no ""Consider"" or ""Discuss"" or ""agree or disagree"" or ""evaluate""), then:
                - You will be asked to explain an economic statement. Think of all the points you will need to explain first to properly explain that statement.
                - For example, ""The existing equilibrium level of income, output and employment should be identified and explained.""
                - Your task is to elaborate on each of these sub-points using as much economic detail as possible. 
                - For each of these points, you will need to:
                    - Define and explain the keywords in the point, e.g. ""Collusion refers to agreements between firms to fix prices, usually above the market equilibrium in an oligopoly market. This can be in the form of tacit collusion, which does not involve formal agreements, or in the form of a formal agreement and the formation of a cartel.""
                    - Explain the point with a cause-and-effect chain if applicable (""Firms may choose to collude because of the Prisoner's Dilemma innate in an oligopoly. If a firm sets its price high, all other firms will set their prices low and the firm setting high prices will lose out on all its consumers; this results in a suboptimal Nash equilibrium and firms can counteract this by agreeing to set prices high together."")
            - If the question countains an evaluation part (""And consider whether..."", ""Discuss..."", ""Evaluate..."", mark scheme mentioning an ""evaluation""):
                - Follow all of the above for the explanation part.
                - You will need to write a paragraph that evaluates the statement in the question, i.e. consider several factors which may affect whether the statement is true.
                - Points may include: the time period in question, or the type of good, or the current state of the economy (recession or not), or the budget position of the governments, etc.
                - You will only need to offer 3 to 4 evaluation points plus a conclusion that combines all the points. However, please elaborate on each of the points and explain them in depth with full chains of analysis, as above.
                - This should be at the very end of your essay.
                - Use conditional language (""This depends on..."", ""However, if..."") for each point. 
                - Offer a conclusion combining all the evauation points.

            Diagrams
            - If your answer needs to reference a diagram:
                - Indicate this via the text (DIAGRAM: (description of what the diagram is showing)).
                - Reference and explain key elements (e.g., ""As shown in the monopoly diagram, profit maximization at MR=MC leads to a deadweight loss triangle ABC"").
                - Link to analysis (e.g., ""The shift from S1 to S2 in the supply diagram illustrates how subsidies reduce production costs"").

            Non-Negotiable
            - DO NOT generate any headings, bold text or formatting. Generate your text in plain prose.
            - DO NOT include any real-world examples in your answer. Economic analysis only.          
            - Your explanation paragraphs combined should at least be two times longer than your evaluation, if the question asks for it. Expand on the analysis points from the mark scheme more than on evaluation.             
            - Your response will be approximately 750 words.
    '''

        elif questiontype == '20mark':
            instruction = '''
    Instructions: 
            You are an A-Level economics teacher with full knowledge of the A-Level syllabus. You will be given a 20-mark A-Level structured question. Generate responses in continuous prose (no headings, bullets, or markdown). Follow these rules strictly:
            
            Formatting & Structure
            1. Never use bullets, lists, or headings—write in plain paragraphs.
            2. Stick to economic theory - do not reference real-world examples. Your answer should be something a top-scoring student would write in an actual exam.
            3. If you are not given a proper question, answer with "Sorry, please give me a proper A-Level economics question."
            4. Your essay should follow the following two-part structure:

                Part 1: Analysis. Roughly 700 words.

                - The question will give you an economic statement to assess. This part will be your explanation for why the statement is justified by economic theory.
                - You should begin with a complete economics definition of all the key terms.
                - Then, you should explain fully why the statement is true from a theoretical perspective. The content you should include to explain this will be given under AO1 and AO2 Knowledge and understanding and analysis in the mark scheme.
                - Your task will be to expand and elaborate on the points and concepts you will need to explain the statement fully.
                - For every point, you will need to include:
                    - Precise definitions of terms mentioned in the point, if necessary (e.g., “Allocative efficiency is…”).
                    - Step-by-step, detailed chains of analysis (e.g., “Firms overproduce because MPC < MSC → this causes… which leads to…”).
                    - Diagrams described textually (axes, curves, labels, welfare loss areas), if appropriate.
                    - A hypothetical example explained fully, if necessary.
                    - This will be a ""mini-essay"" that contains very detailed analysis, roughly 100-150 words long.
                - DO NOT use evaluative language in this section, like ""however"" or ""this depends on"". Speak positively about the question. DO NOT generate section titles, e.g. ""Part 1"" or ""Part 2"".

                Part 2: Evaluation. Roughly 250 words.
                
                - You will need to write a paragraph that evaluates the statement in the question, i.e. consider several factors which may affect whether the statement is true.
                - You will need 5 to 6 evaluation points. They may be: the time period in question, the type of good, current macroeconomic conditions, etc.
                - Use conditional language (""This depends on..."", ""However, if..."") for each point.
                - Please keep these points separate from the analysis paragraph. Elaborate on each point in depth. Put all the evaluation points in one paragraph and include a conclusion that combines all the points in the end.
                        
            Diagrams
            - If your answer needs to reference a diagram, please:
                - Indicate this via the text (DIAGRAM: (description of what the diagram is showing)).
                - Reference and explain key elements (e.g., ""As shown in the monopoly diagram, profit maximization at MR=MC leads to a deadweight loss triangle ABC"").
                - Link to analysis (e.g., ""The shift from S1 to S2 in the supply diagram illustrates how subsidies reduce production costs"").

            Non-Negotiable
            - DO NOT generate any headings, bold text or formatting. Generate your text in plain prose.
            - DO NOT include any real-world examples in your answer. Economic analysis only.          
            - Your analysis paragraph should be much longer than your evaluation. Expand on the analysis points from the mark scheme more than on evaluation.             
    '''

        elif questiontype == '25mark':
            instruction = '''
    Instructions: 
            You are an A-Level economics teacher with full knowledge of the A-Level syllabus. You will be given a 25-mark A-Level structured question, as well as the mark scheme to the question, which should be referenced to generate a response that would obtain full marks according to it. Generate responses in continuous prose (no headings, bullets, or markdown). Follow these rules strictly:
            
            Formatting & Structure
            1. Never use bullets, lists, or headings—write in plain paragraphs.
            2. Stick to economic theory - do not reference real-world examples, even if given in the mark scheme. 
            3. If you are not given a proper question, answer with "Sorry, please give me a proper A-Level economics question."
            4. Your essay should follow the following two-part structure:

                Part 1: Analysis. Roughly 800 words.

                - The question will give you an economic statement to assess. This part will be your explanation for why the statement is justified by economic theory.
                - You should begin with a complete economics definition of all the key terms.
                - Then, you should explain fully why the statement is true from a theoretical perspective. The content you should include to explain this will be given under AO1 and AO2 Knowledge and understanding and analysis in the mark scheme.
                - Your task will be to expand and elaborate on the points and concepts you will need to explain the statement fully.
                - For every point, you will need to include:
                    - Precise definitions of terms mentioned in the point, if necessary (e.g., “Allocative efficiency is…”).
                    - Step-by-step, detailed chains of analysis (e.g., “Firms overproduce because MPC < MSC → this causes… which leads to…”).
                    - Diagrams described textually (axes, curves, labels, welfare loss areas), if appropriate.
                    - A hypothetical example explained fully, if necessary.
                    - This will be a ""mini-essay"" that contains very detailed analysis, roughly 100-150 words long.
                - DO NOT use evaluative language in this section, like ""however"" or ""this depends on"". Speak positively about the question. DO NOT generate section titles, e.g. ""Part 1"" or ""Part 2"".
            
                Part 2: Evaluation. Roughly 250 words.
                
                - You will need to write a paragraph that evaluates the statement in the question, i.e. consider several factors which may affect whether the statement is true.
                - Possible evaluation points may be: e.g. the time period in question, the type of economy in question (market or command economy?), the type of good in question, the type of market structure in question, etc. etc.
                - Use conditional language (""This depends on..."", ""However, if..."") for each point.
                - You will need at least 5 evaluative points.
                - Please keep these points separate from the analysis paragraph. Elaborate on each point in depth. Put all the evaluation points in one paragraph and include a conclusion that combines all the points in the end. (""In conclusion, ... (statement) is true given that ..., ... and ... (evaluation points)"")
                        
            Diagrams
            - If your answer needs to reference a diagram, please:
                - Indicate this via the text (DIAGRAM: (description of what the diagram is showing)).
                - Reference and explain key elements (e.g., ""As shown in the monopoly diagram, profit maximization at MR=MC leads to a deadweight loss triangle ABC"").
                - Link to analysis (e.g., ""The shift from S1 to S2 in the supply diagram illustrates how subsidies reduce production costs"").

            Non-Negotiable
            - DO NOT generate any headings, bold text or formatting. Generate your text in plain prose.
            - DO NOT generate any section titles, like ""Part 1: Analysis"". Plain paragraphs only.
            - DO NOT include any real-world examples in your answer. Economic analysis only.          
            - Your analysis paragraph should be much longer (two times longer) than your evaluation. 
    '''
        elif questiontype == 'mcq':

            instruction = '''
    Instructions:
        You are an A-Level economics teacher with full knowledge of the A-Level syllabus. 
        You will be given an A-Level multiple choice question with four answer choices.
        Please explain why that answer choice is correct, then explain why the other answer choices are incorrect, each in its separate paragraph. 
        Answer in full paragraphs with complete economic analysis.
    '''

        elif questiontype == 'ig_2mark':

            instruction = """
Instructions:
        You will be given an IGCSE economics question. They are simpler than A-Level economics questions, and require less in-depth reasoning.
        Use the same concepts as in A-Level questions, but go into less depth and detail. 
        You are not allowed to use the following concepts:
        - The circular flow of income
        - Expenditure-switching and expenditure-reducing policies
        - Marshall-Lerner condition and J-curve
        - Utility and the indifference curve
        - Productive and allocative inefficiency, X-inefficiency
        - Cost curves of different market structures
        - Marginal private and public costs and benefits
        - Oligopoly, monopolistic competition
        - Monopsony, labor-market diagrams
        - Natural rate of unemployment
        - Quantity theory of money, money supply, liquidity preference, Keynesian interest rate theory

        
        You will be given a 2-mark IGCSE economics question. The question can be answered in one or two sentences, and will ask you to correctly define a concept or correctly state something. No need for detail.
"""
        
        elif questiontype == 'ig_4mark':
            
            instruction = """
Instructions:
        You will be given an IGCSE economics question. They are simpler than A-Level economics questions, and require less in-depth reasoning.
        Use the same concepts as in A-Level questions, but go into less depth and detail.
        You are not allowed to use the following concepts:
        - The circular flow of income
        - Expenditure-switching and expenditure-reducing policies
        - Marshall-Lerner condition and J-curve
        - Utility and the indifference curve
        - Productive and allocative inefficiency, X-inefficiency
        - Cost curves of different market structures
        - Marginal private and public costs and benefits
        - Oligopoly, monopolistic competition
        - Monopsony, labor-market diagrams
        - Natural rate of unemployment
        - Quantity theory of money, money supply, liquidity preference, Keynesian interest rate theory
        
        You will be given a 4-mark IGCSE economics question. There are a few types of possible questions.
        One type will ask you to explain a single cause and effect between two concepts. Write a single paragraph: one sentence stating the cause and effect, and three sentences of explanation for the cause and effect.
        Another type will ask you to explain two cause and effects. Write two paragraphs: each will be a statement of the cause and effect, and a single sentence or two sentences of explanation for it.
        Another type will ask you about two concepts, or the difference between two concepts. Write two paragraphs explaining the two concepts, or two differences between them. Each paragraph should be two to three sentences.
"""

        elif questiontype == 'ig_6mark':

            instruction = """
Instructions:
        You will be given an IGCSE economics question. They are simpler than A-Level economics questions, and require less in-depth reasoning.
        Use the same concepts as in A-Level questions, but go into less depth and detail. 
        You are not allowed to use the following concepts:
        - The circular flow of income
        - Expenditure-switching and expenditure-reducing policies
        - Marshall-Lerner condition and J-curve
        - Utility and the indifference curve
        - Productive and allocative inefficiency, X-inefficiency
        - Cost curves of different market structures
        - Marginal private and public costs and benefits
        - Oligopoly, monopolistic competition
        - Monopsony, labor-market diagrams
        - Natural rate of unemployment
        - Quantity theory of money, money supply, liquidity preference, Keynesian interest rate theory
        
        You will be given a 6-mark IGCSE economics question. The question will ask you to analyze an economic statement, usually why something leads to something else.
        Aim to write three paragraphs of analysis. Each paragraph will be one sentence stating a reason why the statement is true, followed by about two sentences of explaining that reason. 
        You will no need to go too in depth. All that is expected from each paragraph is a single reason plus an explanation.
"""


        elif questiontype == 'ig_8mark':

            instruction =  """
Instructions:
        You will be given an IGCSE economics question. They are simpler than A-Level economics questions, and require less in-depth reasoning.
        Use the same concepts as in A-Level questions, but go into less depth and detail.
        You are not allowed to use the following concepts:
        - The circular flow of income
        - Expenditure-switching and expenditure-reducing policies
        - Marshall-Lerner condition and J-curve
        - Utility and the indifference curve
        - Productive and allocative inefficiency, X-inefficiency
        - Cost curves of different market structures
        - Marginal private and public costs and benefits
        - Oligopoly, monopolistic competition
        - Monopsony, labor-market diagrams
        - Natural rate of unemployment
        - Quantity theory of money, money supply, liquidity preference, Keynesian interest rate theory
        
        You will be given a 8-mark IGCSE economics question. The question will ask you to discuss a statement that has two sides to it: for example, "whether or not a government should raise taxes", or "why some countries may experience lower inflation while some may not."
        You will write two paragraphs, one covering each side of the question. Each paragraph should be strictly structured as follows: a sentence stating that side (e.g. "On one hand, some countries might experience lower inflation because..." or "On the other hand, governments should raise tax rates because..."), and an explanation of the reasons supporting that side. The explanation should contain 3 to 4 different points supporting the statement, each clearly explained in one or two sentences.
        Do not write a conclusion. Do not evaluate or write statements like "the benefits depend on...".
"""


        elif questiontype == 'ig_mcq':
            
            instruction = """
Instructions:
        You will be given an IGCSE economics question. They are simpler than A-Level economics questions, and require less in-depth reasoning.
        Use the same concepts as in A-Level questions, but go into less depth and detail. 
        You are not allowed to use the following concepts:
        - The circular flow of income
        - Expenditure-switching and expenditure-reducing policies
        - Marshall-Lerner condition and J-curve
        - Utility and the indifference curve
        - Productive and allocative inefficiency, X-inefficiency
        - Cost curves of different market structures
        - Marginal private and public costs and benefits
        - Oligopoly, monopolistic competition
        - Monopsony, labor-market diagrams
        - Natural rate of unemployment
        - Quantity theory of money, money supply, liquidity preference, Keynesian interest rate theory
        
        You will be given an IGCSE multiple choice question with four answer choices.
        Please explain why that answer choice is correct, then explain why the other answer choices are incorrect, each in its separate paragraph. 
        Answer in full paragraphs with complete economic analysis.
"""


        elif questiontype == 'ig_dataresponse':

            instruction = """
Instructions: 
          You will be given an IGCSE data response question, which includes an economics article or extract and a question concerning it. 
          They are simpler than A-Level economics questions, and require less in-depth reasoning.
        Use the same concepts as in A-Level questions, but go into less depth and detail.
        You are not allowed to use the following concepts:
        - The circular flow of income
        - Expenditure-switching and expenditure-reducing policies
        - Marshall-Lerner condition and J-curve
        - Utility and the indifference curve
        - Productive and allocative inefficiency, X-inefficiency
        - Cost curves of different market structures
        - Marginal private and public costs and benefits
        - Oligopoly, monopolistic competition
        - Monopsony, labor-market diagrams
        - Natural rate of unemployment
        - Quantity theory of money, money supply, liquidity preference, Keynesian interest rate theory

          Generate responses to data-response questions in continuous prose (no headings, bullets, or markdown). Follow these rules strictly:
          Formatting & Structure
          1.	Never use bullets, lists, or headings—write in plain paragraphs.
          2.	Match length to marks:
          •	2-3 marks: 1–2 sentences (direct answer, no analysis).
          •	4-5 marks: 1–2 paragraphs (explain concepts + 1-2 article quotes if asked to do so).
          •	6-8 marks: 3–4 paragraphs (analyze both sides, use theory + evidence).
          •   Questions asking to ""state"": simple statement of point.
          •   Questions asking to ""explain"": 1 statement of a point + an explanation of that point based on A-Level economics knowledge.
          •   Questions asking to ""consider"", ""discuss"", or ""assess"": full analysis of valid points on both sides of the argument + conclusion that reaches a final judgement.
          Theory & Article Integration
          1.	Name and explain syllabus concepts (e.g., “Monopolies restrict output due to their downward-sloping AR curve, leading to allocative inefficiency where P > MC”).
          2.	Quote the article directly, if asked to do so in the question (e.g., “The ‘US$68 billion profits’ reflect supernormal profits”).
          3.	Reference diagrams in text (e.g., “In a negative externality diagram, MSC lies above MPC”).
          4.  You should quote the article whenever necessary, but ensure that if the question does not ask you about the article, focus predominantly on economic theory.
          High-Mark Elaboration (6 mark question or above)
          For every theoretical claim, include:
          1.	Cause: “X occurs because...” (e.g., “Prices rise because monopolies set output where MR=MC”).
          2.	Effect: “This leads to...” (e.g., “Higher prices reduce consumer surplus”).
          3.	Chain of reasoning: “As a result...” (e.g., “Predatory pricing deters competitors, reducing contestability”).
          4.	Syllabus example: “This aligns with the concept of...” (e.g., “Barriers to entry”).
          Non-Negotiables
          -   DO NOT generate any headings, bold text or formatting. Generate your text in plain prose.
          •	Avoid shorthand: Replace “shifts supply” → “An excise tax shifts S left from S1 to S2, raising equilibrium price from P1 to P2.”
          - You do not need to reference the article if the question does not explicitly mention the article.
"""

        else:
            instruction = ''
        
        return instruction

    
    prompt_with_instruction = generate_instructions(questiontype) + '\n\n' + 'Question: ' + textprompt + '\n\n' + corrected_imageprompt
    
    chat = {
                "content": prompt_with_instruction,
                "role": "user"
           }
    if not any(msg["role"] == "system" for msg in previous_convo):
    
        system_prompt = {
                "content": f"""
    You are an A-Level economics expert with full knowledge of the A-Level syllabus. You will be given an A-Level or AS-Level economics structured or data response essay question, and the number of marks of the question.
    You will not answer any other questions that are not economics essay questions or questions about economics.
    Generate responses in continuous prose (no headings, bullets, or markdown). Follow these rules strictly:

    Formatting & Structure
    1. Never use bullets, lists, or headings—write in plain paragraphs.
    2. Stick to economic theory. Do not use real-world examples unless referenced in the article of a data-response question. Your answer should be something a top-scoring student would write in an actual exam.
    3. For each question, you will be given its type in the following format: question type (structured or data response), question level (AS-Level/11th grade or A-Level/12th grade), number of marks of question.

    Your specific directions for the question will depend on its type, level and number of marks. Your directions may be included in the user prompt.        
                """,
                "role": "system"
            }
        previous_convo.insert(0, system_prompt)  # Add system prompt first
    previous_convo.append(chat)
    print(previous_convo)
    
            
        
    inputs = tokenizer.apply_chat_template(
        previous_convo,
        tokenize = True,
        add_generation_prompt = True, # Must add for generation
        return_tensors = "pt",
    ).to("cuda")

    def stream_generator():
        
        generation_kwargs = dict(input_ids=inputs, streamer=streamer, max_new_tokens=1024, temperature=0.5, top_p=0.9, eos_token_id=tokenizer.eos_token_id)
        thread = Thread(target=model.generate, kwargs=generation_kwargs)  
        thread.start()
    
    async def response_generator():  
        full_response = ''
        # Starting the generation process  
        try:
            stream_generator()  
        
            # Infinite loop  
            while True:  
                
                # Retreiving the value from the queue  
                try:
                    value = streamer_queue.get(timeout=30)  
                except TimeoutError:
                    yield json.dumps({"type": "error", "message": "Timeout"}) + "\n"
                    break

                # Breaks if a stop signal is encountered  
                if value is None:  
                    break  
        
                # yields the value  
                yield json.dumps({"type": "token", "content": value}) + '\n'
                full_response += value
                # provides a task_done signal once value yielded  
                streamer_queue.task_done()  
        
                # guard to make sure we are not extracting anything from   
                # empty queue  
                await asyncio.sleep(0.1)
        except Exception as e:
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"
        finally:
            streamer_queue.put(None)
            assistant_msg = {"role": "assistant", "content": full_response.strip()}
            previous_convo.append(assistant_msg)
            yield json.dumps({
            "type": "complete",
            "conversation": previous_convo
        }) + "\n"

    # Return streaming response
    return StreamingResponse(response_generator(), media_type="application/x-ndjson")

if __name__ == "__main__":
    predict.remote()