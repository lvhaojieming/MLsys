"""Verify extra diagnostic holdouts preserve natural validation and exclude whole documents."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import runpy
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


class HoldoutTest(unittest.TestCase):
    def test_natural_prefix_preserved_and_all_extra_document_chunks_excluded(self):
        tokenizer = types.SimpleNamespace(eos_token_id=99)
        transformers = types.ModuleType('transformers')
        transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *a, **kw: tokenizer)
        preparation = types.ModuleType('prepare_router_training_full')
        preparation.SOURCES = ('c4',)
        preparation.requests = lambda *args: iter([
            ('natural', '0:0', [1], [2]),
            ('natural', '0:2048', [3], [4]),
            ('diagnostic', '1:0', [5], [6]),
            ('diagnostic', '1:2048', [7], [8]),
            ('train', '2:0', [9], [10]),
        ])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'raw_counts.json').write_text(json.dumps([dict(source='c4', training_rows=5)]))
            (root/'excluded.json').write_text('["diagnostic"]')
            output = io.StringIO()
            argv = ['stream_router_raw.py', '--raw-root', str(root), '--tokenizer', 'unused',
                '--valid-per-source', '1', '--exclude-groups', str(root/'excluded.json')]
            with patch.dict(sys.modules, {'transformers': transformers, 'prepare_router_training_full': preparation}), \
                    patch.object(sys, 'argv', argv), patch.dict('os.environ'), contextlib.redirect_stdout(output):
                runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/stream_router_raw.py'), run_name='__main__')
            records = [json.loads(line) for line in output.getvalue().splitlines()]
            samples = [r for r in records if 'id' in r]
            self.assertEqual([(r['group_id'], r['split']) for r in samples], [('natural','valid'), ('train','train')])
            ready = next(r for r in records if r.get('event') == 'ready')
            self.assertEqual(ready['excluded_groups_sha256'], hashlib.sha256(json.dumps(['diagnostic']).encode()).hexdigest())
            self.assertEqual(records[-1]['counts'], {'valid':1, 'train':1})


if __name__ == '__main__':
    unittest.main()
