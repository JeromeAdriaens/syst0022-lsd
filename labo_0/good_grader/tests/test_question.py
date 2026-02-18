import unittest
import json
import nbformat as nbf
import hashlib
import ast
import numpy as np
# from torchvision.datasets import Imagenette
from gradescope_utils.autograder_utils.decorators import weight, number

def get_answers(key):
    hash = hashlib.sha256(key.encode()).digest()
    seed = np.frombuffer(hash, dtype=np.uint32)
    rstate = np.random.RandomState(seed)

    # Question 1 & 5
    numbers = rstate.randint(-50, 50, (2, 2))
    sol_1 = numbers[0].sum()
    sol_5 = numbers[1].sum()

    # Question 3
    shape = rstate.randint(3, 10)

    # Question 4
    quiz_options = ['A', 'B', 'C', 'D']
    quiz_colors = ['#3498db', '#27ae60', '#e74c3c', '#f1c40f']
    rstate.shuffle(quiz_colors)
    quiz_answer = rstate.choice(quiz_options)

    answers = [sol_1,
               shape,
               quiz_answer,
               sol_5]
    return answers

def get_student_answers(notebook_path):
    with open(notebook_path) as f:
        notebook = nbf.read(f, as_version=4)
    # Filter for code cells only
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    cell_of_interest = None
    for cell in reversed(code_cells):
        if not cell.outputs or "text" not in cell.outputs[-1]:
            continue
        cell_of_interest = cell
        break

    if not cell_of_interest:
        raise ValueError("No valid output found in notebook")

    answer_cell_output = cell_of_interest.outputs
    sum = answer_cell_output[-1].text

    # Convert the string representation of the dictionary to an actual dictionary
    student_answers = ast.literal_eval(sum).values()

    return list(student_answers)
    
def get_student_key():
    with open('/autograder/submission_metadata.json') as f:
        metadata = json.load(f)
    return metadata['users'][0]['email']

class QuestionTest(unittest.TestCase):
    def setUp(self):
        self.student_key = get_student_key()
        self.student_answers = get_student_answers('/autograder/source/notebook.ipynb')
        self.answers = get_answers(self.student_key)

    @weight(2.5)
    @number("Q1")
    def test_question_1(self):
        self.assertEqual(int(self.student_answers[0]), self.answers[0])


    @weight(2.5)
    @number("Q2")
    def test_question_2(self):
        self.assertEqual(int(self.student_answers[1]), self.answers[1])
    
    @weight(2.5)
    @number("Q3")
    def test_question_3(self):
        self.assertEqual(self.student_answers[2], self.answers[2])

    @weight(2.5)
    @number("Q4")
    def test_question_4(self):
        self.assertEqual(int(self.student_answers[3]), self.answers[3])
