# Some import we need for our plots
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets

from IPython.display import display, Markdown, clear_output, Image
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from ipywidgets import interactive
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline


def load_example_1(degree):
    np.random.seed(0)
    X = np.sort(5 * np.random.rand(80, 1), axis=0)
    y = np.sin(X).ravel() + 0.7 * np.random.randn(80)  
    model = Ridge(alpha=1e-2)
    X_poly = np.vander(X.ravel(), degree + 1, increasing=True)
    
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=0)
    
    model.fit(X_train, y_train)
    
    train_error = np.mean((y_train - model.predict(X_train))**2)
    val_error = np.mean((y_test - model.predict(X_test))**2)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X, y, color='blue', label='Data')
    x_range = np.linspace(0, 5, 100)
    y_pred = model.predict(np.vander(x_range, degree + 1, increasing=True))
    plt.plot(x_range, y_pred, color='red', linewidth=2, label='Model')
    plt.title(f'Polynomial Degree = {degree}')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    
    degrees = range(1, degree + 1)
    train_errors = []
    val_errors = []
    for d in degrees:
        X_poly_d = X_poly[:, :d]
        X_train_d = X_train[:, :d]
        X_test_d = X_test[:, :d]
        
        model.fit(X_train_d, y_train)
        
        train_errors.append(np.mean((y_train - model.predict(X_train_d))**2))
        val_errors.append(np.mean((y_test - model.predict(X_test_d))**2))
    
    plt.subplot(1, 2, 2)
    plt.plot(degrees, train_errors, marker='o', label='Training Error', color='green')
    plt.plot(degrees, val_errors, marker='o', label='Validation Error', color='purple')
    
    if degree >= 7:
        classification = 'Overfitting'
        color = 'red'
    elif degree <= 2:
        classification = 'Underfitting'
        color = 'blue'
    else:
        classification = 'Good Fit'
        color = 'green'
        
    plt.annotate(classification, xy=(degree, val_errors[degree-1]), xytext=(degree, val_errors[degree-1]+0.1),
                 arrowprops=dict(facecolor=color, shrink=0.05), fontsize=12, color=color)
    
    plt.title('Training vs. Validation Error')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    

def load_example_2():
    np.random.seed(0)
    X = np.sort(5 * np.random.rand(80, 1), axis=0)
    y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

    X_train, X_test = X[:60], X[60:]
    y_train, y_test = y[:60], y[60:]

    degrees = [1, 4, 15]
    plt.figure(figsize=(15, 5))

    for i, degree in enumerate(degrees):
        ax = plt.subplot(1, len(degrees), i + 1)

        poly_features = PolynomialFeatures(degree=degree)
        X_poly_train = poly_features.fit_transform(X_train)

        model = LinearRegression()
        model.fit(X_poly_train, y_train)

        y_train_pred = model.predict(X_poly_train)

        X_poly_test = poly_features.transform(X_test)
        y_test_pred = model.predict(X_poly_test)

        plt.scatter(X_train, y_train, label='Training Data')
        plt.plot(X_train, y_train_pred, label='Model Prediction', color='r')

        mse_train = mean_squared_error(y_train, y_train_pred)
        mse_test = mean_squared_error(y_test, y_test_pred)
        plt.title(f'Degree {degree}\nMSE (Train): {mse_train:.2f}, MSE (Test): {mse_test:.2f}')
        plt.legend()

    plt.tight_layout()
    plt.show()    
    
    
    
def load_example_3():
    penguins = sns.load_dataset("penguins")
    penguins = penguins.dropna()  
    X = penguins[['bill_length_mm', 'bill_depth_mm']].values
    y = penguins['species'].astype('category').cat.codes.values  

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    k_values = [1, 5, 100]
    classifiers = []

    for k in k_values:
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(X_train, y_train)
        classifiers.append(clf)

    plt.figure(figsize=(12, 8))

    for i, k in enumerate(k_values):
        plt.subplot(2, 2, i+1)
        clf = classifiers[i]
        xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 500),
                             np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 500))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.4)
        plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.coolwarm, s=20, edgecolor='k')
        plt.title(f'k-NN (k={k})')

    plt.tight_layout()
    plt.show()

    
def load_quiz():
    questions = [
    ("**Question 1:** What is the unavoidable error?", [
        "It measures the amount to which the label y varies at  𝑥0", 
        "variance of the model at  𝑥0 and represents the variability of predictions"
    ], "It measures the amount to which the label y varies at  𝑥0"),
    ("**Question 2:** Which of the two images shows overfitting?", ["Image 1", "Image 2",], "Image 1"),
    ("**Question 3:** Underfit models are too complex to represent the underlying relationships in the data", [True, False], 
     f"{False} Underfit models are too simple to represent the underlying relationships in the data."),
    ("**Question 4:** What does the bias-variance tradeoff refer to in machine learning?", 
     ["Balancing the tradeoff between model size and computation time.", "Balancing the tradeoff between model complexity and model error.",], 
     "Balancing the tradeoff between model complexity and model error."),
    ("**Question 5:** Overfitting occurs when a model:", ["Has high bias", "Fails to generalize to new, unseen data",], 
     "Fails to generalize to new, unseen data"),
    ("**Question 6:** What is the primary purpose of a validation set when training a machine learning model?", 
     ["To provide additional training data to improve model accuracy.", "To test the model's performance on data it has never seen during training.",
     "To train the model with more iterations."], "To test the model's performance on data it has never seen during training."),
    ("**Question 7:** How does increasing the complexity of a machine learning model (e.g., adding more features or layers) typically affect the risk of overfitting?", 
     [" It increases the risk of overfitting.", "It has no impact on overfitting.","It makes the model more interpretable."], "It increases the risk of overfitting."),
    # Add more questions here
]

    # Image paths
    image_paths_q2 = ["overfitting.png", "underfitting.png"]

    # Create a list of ToggleButtons for each question
    buttons = [widgets.ToggleButtons(options=options) for _, options, _ in questions]

    # Create a function to check the answers and display feedback
    def check_answers(buttons):
        global all_correct  # Declare all_correct as a global variable
        all_correct = True
        feedback = []

        for i, (question_text, options, correct_answer) in enumerate(questions):
            display(Markdown(question_text))

            # Display image below the question for Question 2
            if i == 1:
                display(Image(filename=image_paths_q2[0], width=200), Image(filename=image_paths_q2[1], width=200)),   # Adjust the width as needed

            # Display the ToggleButtons widget for each question
            toggle_buttons = buttons[i]
            toggle_buttons.style.button_width = 'auto'  # Set the button width to auto
            display(toggle_buttons)

        submit_button = widgets.Button(description="Submit Answers")
        display(submit_button)

        def on_submit_button_click(b):
            clear_output(wait=True)
            global all_correct  # Declare all_correct as a global variable
            for i, (question_text, options, correct_answer) in enumerate(questions):
                display(Markdown(question_text))

                # Display image below the question for Question 2
                if i == 1:
                    display(Image(filename=image_paths_q2[0], width=200), Image(filename=image_paths_q2[1], width=200))  # Adjust the width as needed

                user_answer = buttons[i].value
                if user_answer == correct_answer:
                    feedback.append(f"\n**Q{i+1}: Correct!** {correct_answer}")
                else:
                    feedback.append(f"\n**Q{i+1}: Incorrect.** The correct answer is {correct_answer}.")
                    all_correct = False
            if all_correct:
                feedback.append("**All answers are correct!**")
            display(Markdown('\n'.join(feedback)))

        submit_button.on_click(on_submit_button_click)

    # Call the check_answers function to start the quiz
    check_answers(buttons)

    
    
    
