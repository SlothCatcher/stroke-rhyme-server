import unittest
from app import Request, rhymes, transcribe

class RhymeServerTests(unittest.TestCase):
    def test_russian_model_groups(self):
        labels = rhymes(Request(terms=['река́', 'дом', 'рука́', 'гром']))['labels']
        self.assertEqual(labels[0], labels[2])
        self.assertEqual(labels[1], labels[3])
        self.assertNotEqual(labels[0], labels[1])
        self.assertIsNotNone(labels[0])

    def test_manual_stress_used_in_transcription(self):
        first, unknown = transcribe('за́мок')
        second, unknown2 = transcribe('замо́к')
        self.assertFalse(unknown or unknown2)
        self.assertNotEqual(first, second)
        self.assertIn('ˈ', first)

if __name__ == '__main__':
    unittest.main()
