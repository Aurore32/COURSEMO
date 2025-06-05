import unicodeitplus as ui

print(ui.parse('''The observed frequency is lower than the source frequency, indicating the car is moving away. Using the Doppler effect formula for a receding source:  
\[ f' = \frac{f v}{v + v_s} \]  
Substituting \( f' = 360 \, \text{Hz} \), \( f = 400 \, \text{Hz} \), and \( v = 340 \, \text{ms}^{-1} \):  
\[ 360 = \frac{400 \times 340}{340 + v} \]  
Solving gives \( v = 37.8 \, \text{ms}^{-1} \approx 38 \, \text{ms}^{-1} \). The direction is away from the observer.'''))