#!/usr/bin/env python3
"""Backport the installed upstream V1 prompt-logprob path to Ascend 0.8.4.

This does not replace sampling or normalization. The native V1 sampler computes
and gathers logprobs on the NPU; only compact output tensors return to the CPU.
"""
import ast
from pathlib import Path
import os


def patch(ascend_path, upstream_path):
    target = Path(ascend_path)
    text = target.read_text()
    marker = '# MOQE_NATIVE_V1_PROMPT_LOGPROBS'
    mask_marker = '# MOQE_NATIVE_V1_PARTIAL_PREFILL_MASK'
    if marker in text and mask_marker in text:
        return False
    if marker not in text:
        upstream = Path(upstream_path).read_text()
        tree = ast.parse(upstream)
        method = next(node for node in ast.walk(tree)
                      if isinstance(node, ast.FunctionDef) and node.name == '_get_prompt_logprobs_dict')
        lines = upstream.splitlines(keepends=True)
        implementation = ''.join(lines[method.lineno - 1:method.end_lineno])
        implementation = implementation.replace('torch.cuda.synchronize()', 'torch.npu.synchronize()')
        replacements = [
            ('from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput',
             'from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput, LogprobsTensors'),
            ('        cu_num_tokens = np.cumsum(num_scheduled_tokens)',
             '        cu_num_tokens = np.cumsum(num_scheduled_tokens)\n'
             '        self.query_start_loc_np = np.concatenate((np.zeros(1, dtype=cu_num_tokens.dtype), cu_num_tokens))'),
            ('        return hidden_states[sample_indices]',
             '        self._native_prompt_hidden_states = hidden_states\n'
             '        return hidden_states[sample_indices]'),
            ('    def apply_grammar_bitmask(', implementation + '\n\n    def apply_grammar_bitmask('),
            ('        # TODO(woosuk): The following loop can be slow since it iterates over',
             '        prompt_logprobs_dict = self._get_prompt_logprobs_dict(\n'
             '            self._native_prompt_hidden_states, scheduler_output)\n'
             '        del self._native_prompt_hidden_states\n\n'
             '        # TODO(woosuk): The following loop can be slow since it iterates over'),
            ('            prompt_logprobs_dict={},', '            prompt_logprobs_dict=prompt_logprobs_dict,'),
        ]
        for old, new in replacements:
            if text.count(old) != 1:
                raise RuntimeError('Unsupported Ascend source; expected exactly one occurrence: ' + old)
            text = text.replace(old, new)
        text = marker + '\n' + text
    mask_replacements = [
        ('        # TODO(woosuk): The following loop can be slow since it iterates over',
         '        # MOQE_NATIVE_V1_PARTIAL_PREFILL_MASK\n'
         '        discard_sampled_tokens_req_indices = []\n'
         '        # TODO(woosuk): The following loop can be slow since it iterates over'),
        ('                    generator.set_offset(generator.get_offset() - 4)',
         '                    generator.set_offset(generator.get_offset() - 4)\n'
         '                discard_sampled_tokens_req_indices.append(i)'),
        ('        model_runner_output = ModelRunnerOutput(',
         '        for i in discard_sampled_tokens_req_indices:\n'
         '            valid_sampled_token_ids[i].clear()\n\n'
         '        model_runner_output = ModelRunnerOutput('),
    ]
    for old, new in mask_replacements:
        if text.count(old) != 1:
            raise RuntimeError('Unsupported partial-prefill source: ' + old)
        text = text.replace(old, new)
    ast.parse(text)
    backup = target.with_suffix('.py.before-native-prompt-logprobs')
    if not backup.exists():
        backup.write_text(target.read_text())
    temporary = target.with_suffix('.py.native-tmp')
    temporary.write_text(text)
    os.replace(temporary, target)
    return True


if __name__ == '__main__':
    import fcntl
    with Path('/workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py.native-prompt-logprobs.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        changed = patch('/workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py',
                        '/workspace/vllm/vllm/v1/worker/gpu_model_runner.py')
        attention = Path('/workspace/vllm-ascend/vllm_ascend/attention/attention.py')
        source = attention.read_text()
        if '.mask_fill_(' in source:
            fixed = source.replace('.mask_fill_(', '.masked_fill_(')
            ast.parse(fixed)
            backup = attention.with_suffix('.py.before-masked-fill-fix')
            if not backup.exists():
                backup.write_text(source)
            temporary = attention.with_suffix('.py.native-tmp')
            temporary.write_text(fixed)
            os.replace(temporary, attention)
    print('Native V1 prompt-logprob compatibility:', 'installed' if changed else 'already installed')
