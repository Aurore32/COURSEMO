from openai import OpenAI
import pandas as pd


client = OpenAI(api_key="sk-61dd5e4c06c4450893eb429ccf00e0fd", base_url='https://api.deepseek.com')
responses = []

df = pd.read_csv('./physics/distillation/al_physics_experiment_clean.csv')
df = df.rename(columns={'Question': 'Answer', 'Answer': 'Question'})
questions = df['Question'].tolist()
markschemes = df['Answer'].tolist()
actual_questions = []
actual_markschemes = []
for i in range(len(questions)):
    question = questions[i]
    markscheme = markschemes[i]
    print(question)
    print(markscheme)
    response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "system", "content": """
                You are an A-Level Physics expert teacher with full knowledge of the A-Level Physics syllabus. 
                You are trying to write model answers for physics questions for your students to learn from.
                You will be given an A-Level Physics experiment planning question, worth 15 marks, and its associated marking scheme. Strictly follow these rules:
                - Generate only the answer. Write nothing else.
                - If there is a figure in the question, ignore it.
                - Write in plain prose only. Do not generate section titles, headings, or bold text.
                - Your task is to organize the points of the marking scheme into a coherent model answer that would obtain full marks according to it.
                - You will elaborate greatly and in detail on each point written in the mark scheme, expanding them far beyond what they originally state.
                - The question will concern an experiment to verify a physical equation which involves several variables. The answer will be in the following four sections, each being headed by a bold title:
                    - **Defining the problem**, worth 2 marks: a statement of the independent variable and the dependent variable. A statement of the variables that you will need to keep controlled.
                    This will usually be any variable in the equation given that is not the independent and dependent variable. List every single variable that should be controlled.
                        - Check the mark scheme under "defining the problem".
                    - **Methods of data collection**, worth 4-5 marks:
                        - Check the mark scheme under "methods of data collection".
                        - For every variable in the equation, state exactly how to measure it in great detail: which variable it is, its units, what apparatus is used to measure it and how it is measured with that apparatus.
                        You should use the most accurate apparatus available in a typical school lab (e.g. a micrometer caliper is preferred over a ruler).
                        - At the end of this section, describe a diagram that suggests how to conduct the experiment using your apparatus given. Provide this description in a separate line: (DIAGRAM: (description of diagram for experiemnt)).
                    - **Methods of analysis**, worth 4-5 marks:
                        - Check the mark scheme under "methods of analysis".
                        - You will be required to describe how to use data from the experiment to prove or disprove the equation given
                        - The question will tell you "it is suggested that (dependent variable) is related to (independent variable) by the relationship ..."
                        - You will write out the algebra needed to transform the equation given into a form which can be used to plot a linear regression between the independent and dependent variable
                        - This may be in the form of a power or a logarithm of either of the two variables
                        - Then, the question will ask you how this equation can be used to determine values for constants in the equation. The constants will be related to the y-intercept and gradient of the linear equation. Describe algebraically the relationship between these constants and the y-intercept or the gradient.
                        - If the mark scheme and your calculations come into conflict, use your calculations. The mark scheme may be inaccurate.
                    - **Additional detail including safety considerations**, worth 4-6 marks:
                        - Check the mark scheme under "additional detail" for valid points.
                        - Provide as many points as possible     on how the experiment could be improved or made more accurate with apparatus available in a school lab
                        - One point is always: "The relationship in the equation is valid if the data is a straight line (either through the origin if there is no y-intercept, or not through the origin if there is a y-intercept, depending on what the equation given is)".
                        - Another common point is to average the dependent variable across 3 or more trials.
                        - One mark will always be awarded for a safety procedure, e.g. wearing gloves to protect against heat or wearing goggles. 
                        - Marks may be awarded for stating all the variables to be held constant which were not stated before.
                        - The other points can be ways to improve the measurements, or details on how the measurements are made:
                            - For instance, using a fiducial marker to mark the original length of a spring. 
                            - For instance, an elaboration on how resistance is measured: a circuit connected with a voltmeter and an ammeter.
                        - Go into great detail (2 sentences or more) on each of those points.
                        """},
    {
        'role': 'user', 'content': f"""
    Question:
    A trolley with a magnet attached is placed on a thin steel sheet as shown in Fig. 1.1.

(Fig 1.1)

The angle between the sheet and the bench is θ. The distance from point X to the trolley is d.
The trolley is released from rest and travels down the slope. The velocity v of the trolley at X is
determined using a light gate.
It is suggested that v is related to θ by the relationship
mpsinθ – qB =
mv2
2d
where m is the mass of the trolley and magnet, B is the magnetic flux density between the magnet
and the steel sheet, and p and q are constants.
Plan a laboratory experiment to test the relationship between v and θ.
Draw a diagram showing the arrangement of your equipment.
Explain how the results could be used to determine values for p and q.
In your plan you should include:
• the procedure to be followed
• the measurements to be taken
• the control of variables
• the analysis of the data
• any safety precautions to be taken.

    Mark Scheme:

    Defining the problem
θ is the independent variable and v is the dependent variable, or vary θ and measure v. 1
 Keep d constant 1

Methods of data collection
Labelled diagram of workable experiment including:
• sheet supported by stand / jack
• light gate positioned at X
• support, light gate and X labelled. 1
 Light gate connected to timer / datalogger. 1
 Measure length (L) (of card) interrupted by beam for single light gate. 1
Method to measure θ, e.g. use protractor
or
Method to determine θ , e.g. use a rule(r) to measure two appropriate distances to use in a trigonometrical ratio 1

Method of Analysis
Plots a graph of v^2 on y-axis and sin θ on x-axis.
Allow other valid graphs, e.g. sin θ against v^2 
Do not accept log graphs.
1
p = gradient / 2d 1
q = - m * y-intercept / 2Bd for v^2 against sin θ

Additional detail including safety considerations
Any six from:
6

D1 Method to stop the trolley once the trolley passes X, e.g. place a block / stop on the bench near the end of the sheet
Ignore trolley falls

D2 Keep B and m constant
D3 Use a rule(r) to measure d
D4 Method to keep d constant, e.g. mark distance d on the sheet or the starting position of the trolley on the sheet
D5 Method to measure mass of trolley (and magnet), e.g. use balance or use newton meter to measure weight and divide by g
and
Measure B using a (calibrated) Hall probe

D6 Additional detail on use of Hall probe, e.g.
adjust probe until maximum value or
measure B using Hall probe first in one direction, then in the opposite direction and average
D7 Determine
v (the velocity at X) from
L / t (for a single light gate)
D8 Additional detail on measuring
θ, e.g. protractor drawn in correct position on diagram, or
additional detail on determining
θ , e.g. relationship between measured lengths and θ
    """
    },
    {'role': 'assistant',
     'content':
     r"""
**Defining the problem**
In this experiment, the angle $\theta$ of the steel sheet is the independent variable to be changed, and the velocity v of the trolley at X is the dependent variable which is to be measured. The distance from point X to the trolley, alongside the combined mass of the trolley and the magnet m and the magnetic flux density of between the magnet and the steel sheet B, are to be kept constant.
**Methods of data collection**
First, decide on a value for $\theta$ (e.g. 20 degrees); calibrate and confirm that value either by using a protractor to measure the angle made by the slope, or by measuring the height and length of the slope and calculating the arctangent of their ratios to obtain the angle through the trigonometric relation $\theta = \arctan (y/x)$.
The mass of the trolley and magnet combined can be measured via a top-pan balance or a digital scale. Measure d, the distance from point X to the trolley, with a meter rule. Measure the magnetic flux density of between the magnet and the steel sheet with a calibrated Hall probe. 
Place a very small card on the trolley. Place a light gate with a beam facing the card on the trolley to a timer or a data logger beside the steel sheet at point X. For each value of $\theta$ chosen, measure the velocity by measuring the length of the card placed on the trolley L, then the duration of time t in which the beam of the light gate intercepted the card. An estimate of the velocity of the trolley at X, v, will be obtained by v = L / t. 
(DIAGRAM: A steel sheet supported at an angle $\theta$ by a boss and clamp, labelled; a protractor placed perpendicular to the table and parallel to the sheet. A light gate connected to a data logger positioned at X, which should be labelled as well.)
**Methods of Analysis**
To confirm the suggested relationship, plot a graph of $v^2$ on the y-axis against $\sin \theta$ on the x-axis. We have
$$
\frac{2d}{m}(mp \sin \theta – qB) = v^2
$$
And so
$$
2dp\sin \theta - \frac{2dqB}{m} = v^2
$$
Meaning that $p$ is the gradient of the graph divided by $2d$, and $q$ is the $y$-intercept of the graph multiplied by $-\frac{m}{2dB}$.
**Additional detail including safety considerations**
The suggested relationship is valid if a straight line is produced, not necessarily through the origin. Several trials should be conducted, ideally more than 3, for each $\theta$; measurements of $v$ should be averaged from these trials for more accurate data.
For safety reasons, use a sand box or a cushion to catch the trolley if it falls beyond the steel sheet; alternatively, place a block at the end of the sheet to prevent it from falling any further.
An estimate of the velocity of the trolley at X, v, will be obtained by v = L / t, where L represents the length of the card attached to the trolley and t represents the total duration of time elapsed while the light gate’s beam intersects the card. 
Keep all variables aside from $\theta$ and $v$ in the equation constant: in particular, the magnetic flux density of the magnet $B$ and the mass of the trolley and the magnet $m$ are to be kept constant.
Calibrate the Hall probe used to measure $B$ as follows: first measure $B$ by moving the Hall probe in one direction, then measure $B$ again by moving in the other direction. Aggregate the values by averaging them to produce a more accurate reading of $B$. Alternatively, angle the Hall probe and adjust its orientation until a maximum reading is obtained.
Use a fiducial marker to keep $d$ constant; mark in red the position of X and the position of the trolley’s starting position, and ensure that the trolley always starts at the marked position.

"""
     },
    {
'role': 'user',
'content': f"""
Question: {question}

Mark Scheme: {markscheme}   
"""
    }
    ],
            stream=False,
            temperature=0.5
        )
    print(response.choices[0].message.content)
    actual_questions.append(question)
    responses.append(response.choices[0].message.content)
    dataframe = pd.DataFrame([])
    dataframe['Question'] = actual_questions
    dataframe['Response'] = responses
    dataframe.to_csv('./physics/distillation/al_physics_deepseek_r1_experiment.csv')