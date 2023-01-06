# Baby-Cry-Analysis-System
CRYING is the primary means of communication between the baby and the outside world.Infant cry recognition is a challenging task as it is hard to determine between different types of cries. However, baby cry is treated as a different way of communication of speech.So we proposed an automatic infant cry classification model that can be addressed as a better solution.
An infant cry audio corpus has been built through the Donate-a-cry campaign and that database is used for the project.The Dataset is comprised of continuous signals which are preprocessed,filtered,normalized and cleaned.The pre-processed signal is used to extract features using MFCC algorithm where frame blocking,windowing are performed and the co-efficient values are generated.The co-efficient values are given to the different classifiers  and the corresponding co-efficient values are converted to probability distribution values.We use different types of classifiers like SVM, logistic regression,Random forest,decision tree for the cry analysis and the accuracy level of those classifiers are compared.
Applications:
Baby remote monitor-Helpful for the parents to know the reasons behind their baby cry.
Used to classify the cries of the babies in a noisy environment and to improve the overall baby cry recognition rate.
Help inform the parents when their child is hungry,pain,discomfort
 Non-intrusive phychological research of infants and their caregivers in the earliest days of life
 Important for researchers, who study the relation between baby cry patterns and various health or developmental parameters
Decision tree has shown the highest accuracy among all the classifiers with an accuracy rate of 97%.
Decision tree is used effectively for audio signal analysis
More than 90% of the actual output is matched with the predicted output of the machine learning model.
Every other reason for baby crying can also be identified in the future.
 As there is no adequate database ,the primary goal is to collect a very large database consisting of cries of infants from 0 to 9 months old
Combining new audio signal processing methods and novel machine learning methods will lead to a remarkable future.


