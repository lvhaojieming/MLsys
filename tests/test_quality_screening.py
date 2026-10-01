"""Check rejection of demonstrably bad content and safe arithmetic parsing."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    'quality_screening', Path(__file__).parents[1] / 'scripts/refine_router_quality.py')
quality = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quality)


class QualityScreeningTests(unittest.TestCase):
    def test_arithmetic_is_exact_and_does_not_execute_calls(self):
        self.assertEqual(quality.fraction_eval('0.1 + 0.2'), quality.fraction_eval('0.3'))
        with self.assertRaises(ValueError):
            quality.fraction_eval('__import__("os").system("false")')

    def test_wrong_annotation_and_final_are_rejected(self):
        r = {'source': 'gsm8k', 'messages': [{'role': 'user', 'content': 'How much is left?'}],
             'reference_answer': 'We begin with 10 dollars. Subtract the 2 dollar expense: <<10-2=8>>8 dollars are left.\n#### 8',
             'target_tokens': 50}
        self.assertIsNone(quality.screen(r)[0])
        bad = dict(r, reference_answer=r['reference_answer'].replace('10-2=8', '10-2=9'))
        self.assertEqual(quality.screen(bad)[0], 'incorrect_annotated_arithmetic')
        bad = dict(r, reference_answer=r['reference_answer'].replace('#### 8', '#### 7'))
        self.assertEqual(quality.screen(bad)[0], 'final_not_equal_last_calculation')

    def test_python_syntax_is_checked_without_executing_code(self):
        r = {'source': 'Magicoder-OSS-Instruct-75K',
             'messages': [{'role': 'user', 'content': 'Create a Python function that returns the sum of the elements of an integer list, handles an empty list by returning zero, and does not change its input.'}],
             'reference_answer': '```python\ndef total(values):\n    return sum(values)\n```\nThis returns zero for an empty input and does not mutate the input list.',
             'target_tokens': 45, 'metadata': {'code_language': 'python'}}
        self.assertIsNone(quality.screen(r)[0])
        bad = dict(r, reference_answer=r['reference_answer'].replace('return sum(values)', 'return sum('))
        self.assertEqual(quality.screen(bad)[0], 'python_syntax_error')

    def test_short_requests_and_refusals_are_not_high_quality(self):
        r = {'source': 'WildChat-1M', 'messages': [{'role': 'user', 'content': '1234'}],
             'reference_answer': 'Here is a sufficiently long answer that nevertheless has no meaningful request to address and should be excluded.',
             'target_tokens': 30, 'prompt_tokens': 16, 'metadata': {'language': 'English'}}
        self.assertEqual(quality.screen(r)[0], 'low_information_request')
        r['reference_answer'] = "I'm sorry, I cannot understand the question and cannot provide a meaningful answer. Please clarify your request."
        self.assertEqual(quality.screen(r)[0], 'refusal_or_apology')


if __name__ == '__main__':
    unittest.main()
