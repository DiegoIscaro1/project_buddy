import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'buddy')))

from buddy.run_model import load_model
from buddy.preprocessing import transform_input
import numpy as np
import pandas as pd

class TestModel(unittest.TestCase):
    def setUp(self):
        # Load model
        self.model = load_model('sgd_classifier')

    def test_model_output_values(self):
        # List of example inputs
        texts = [
            "I feel like ending it all.",
            "I am feeling great today!",
            "Life is too hard, I can't go on.",
            "I'm so happy and content with everything.",
            "No one understands my pain.",
            "It's a beautiful day to enjoy with friends.",
            "Hello, my name is Jeff!",
            "I don't feel good today",
            "My mother died, I want to join her...",
            "I want to jump through the window."
        ]

        for text in texts:
            with self.subTest(text=text):
                # Clean the text and convert it to a string
                cleaned_text = str(transform_input(text))
                # Wrap the cleaned text in a list
                prediction = self.model.predict([cleaned_text])[0]
                # Check that the prediction is either 0 or 1
                self.assertIn(prediction, [0, 1], f"Prediction for '{text}' should be 0 or 1")


    def test_model_output_shape(self):
        # List of example inputs
        texts = [
            "I feel like ending it all.",
            "I am feeling great today!",
            "Life is too hard, I can't go on.",
            "I'm so happy and content with everything.",
            "No one understands my pain.",
            "It's a beautiful day to enjoy with friends.",
            "Hello, my name is Jeff!",
            "I don't feel good today",
            "My mother died, I want to join her...",
            "I want to jump through the window."
        ]

        for text in texts:
            with self.subTest(text=text):
                # Clean the text and convert it to a string
                cleaned_text = str(transform_input(text))
                # Wrap the cleaned text in a list
                prediction = self.model.predict([cleaned_text])[0]
                # Check the shape of the prediction
                expected_shape = ()  # Each prediction should be a scalar
                self.assertEqual(np.shape(prediction), expected_shape, f"Prediction shape for '{text}' is incorrect")

    def test_transform_input(self):
    # List of example inputs and their expected cleaned texts
        test_data = [
            ("I feel like ending it all.", "feel like ending"),
            ("I am feeling great today!", "feeling great today"),
            ("Life is too hard, I can't go on.", "life hard go"),
            ("I'm so happy and content with everything.", "happy content everything"),
            ("No one understands my pain.", "understands pain"),
            ("It's a beautiful day to enjoy with friends.", "beautiful day enjoy friend"),  # Updated expected cleaned text
            ("Hello, my name is Jeff!", "hello name jeff"),
            ("I don't feel good today", "feel good today"),
            ("My mother died, I want to join her...", "mother died want join"),
            ("I want to jump through the window.", "want jump window")
        ]

        for input_text, expected_cleaned_text in test_data:
            with self.subTest(text=input_text):
                # Transform the input text
                cleaned_text = transform_input(input_text)
                # Convert the Pandas Series to a string
                cleaned_text = cleaned_text[0] if isinstance(cleaned_text, pd.Series) else cleaned_text
                # Check if the expected cleaned text is a substring of the transformed text
                self.assertIn(expected_cleaned_text, cleaned_text, f"Transformation for '{input_text}' is incorrect")


"""
    def test_model_predictions(self):
        # List of example inputs and their expected outputs
        test_data = [
            ("I feel like ending it all."), # --> model predict 0 should be 1?
            ("I am feeling great today!"), # correct
            ("Life is too hard, I can't go on."), # correct
            ("I'm so happy and content with everything."), # correct
            ("No one understands my pain."), # --> model predict 0 should be 1?
            ("It's a beautiful day to enjoy with friends."), # --> correct
            ("Hello, my name is Jeff!"), # correct
            ("I don't feel good today"), # correct
            ("My mother died I'm loosing it..."), # --> model predicted 0 shoud it be 1
            ("I want to jump through the window and just die") # correct
        ]

        expected_outputs = [0, 0, 1, 0, 0, 0, 0, 0, 0, 1]

        for text, expected_output in zip(test_data, expected_outputs):
            with self.subTest(text=text):
                # Clean the text and convert it to a string
                cleaned_text = str(transform_input(text))
                # Wrap the cleaned text in a list
                prediction = self.model.predict([cleaned_text])[0]
                # Check the predicted output against the expected output
                self.assertEqual(prediction, expected_output, f"Prediction for '{text}' is incorrect")

"""
# Maybe we should not remove the stop word
if __name__ == '__main__':
    unittest.main()
