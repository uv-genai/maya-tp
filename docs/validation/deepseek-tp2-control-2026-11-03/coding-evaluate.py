#!/usr/bin/env python3
"""Evaluate four fixed coding probes in a filesystem- and network-isolated bwrap."""
import json
import re
import resource
import subprocess
import sys
import tempfile

CASES = {
    'merge_intervals': {
        'function': 'merge_intervals',
        'tests': [
            ([], []),
            ([[1, 3], [2, 6], [8, 10], [10, 12]], [[1, 6], [8, 12]]),
            ([[5, 7], [1, 2], [2, 4]], [[1, 4], [5, 7]]),
            ([[-5, -3], [-4, 0], [7, 7]], [[-5, 0], [7, 7]]),
            ([[1, 10], [2, 3], [4, 6]], [[1, 10]]),
        ],
    },
    'balanced': {
        'function': 'balanced',
        'tests': [
            ('', True), ('a(b[c]{d})', True), ('([)]', False),
            ('{[()]}x', True), (']', False), ('((a)', False),
        ],
    },
    'find_subsequence': {
        'function': 'find_subsequence',
        'tests': [
            (('abcabc', 'cab'), 2), (('aaaaa', 'aaa'), 0),
            (('xyz', ''), 0), (('', 'a'), -1), (('abcd', 'de'), -1),
            (('ababa', 'aba'), 0),
        ],
    },
    'run_length': {
        'function': 'run_length',
        'tests': [
            ('', []), ('aaabbc', [['a', 3], ['b', 2], ['c', 1]]),
            ('ababa', [['a', 1], ['b', 1], ['a', 1], ['b', 1], ['a', 1]]),
            ('  xx ', [[' ', 2], ['x', 2], [' ', 1]]), ('zzzz', [['z', 4]]),
        ],
    },
}


def extract(text):
    match = re.search(r'```(?:python|py)?\s*\n(.*?)```', text, re.I | re.S)
    return match.group(1) if match else text.strip()


def limit():
    resource.setrlimit(resource.RLIMIT_CPU, (3, 3))
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024,) * 2)
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024,) * 2)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def evaluate(name, answer):
    cfg = CASES[name]
    code = extract(answer)
    if len(code.encode()) > 65536:
        return {'error': 'code exceeds 64 KiB'}
    program = '''import json\nimport copy\nnamespace = {}\nsource = %r\ntry:\n    exec(compile(source, "<candidate>", "exec"), namespace)\nexcept BaseException as exc:\n    print("__CODING_RESULT__" + json.dumps({"load_error": type(exc).__name__ + ": " + str(exc)}))\n    raise SystemExit(0)\nfn = namespace.get(%r)\ntests = %r\nresults = []\nfor argument, expected in tests:\n    try:\n        if %r == "find_subsequence":\n            actual = fn(*argument)\n        else:\n            actual = fn(argument)\n        results.append(actual == expected)\n    except BaseException:\n        results.append(False)\nif %r == "merge_intervals":\n    original = [[5, 7], [1, 3], [2, 6]]\n    snapshot = copy.deepcopy(original)\n    try:\n        fn(original)\n        results.append(original == snapshot)\n    except BaseException:\n        results.append(False)\nprint("__CODING_RESULT__" + json.dumps({"cases": results}))\n''' % (code, cfg['function'], cfg['tests'], name, name)
    cmd = ['bwrap', '--unshare-all', '--die-with-parent', '--new-session', '--clearenv',
           '--ro-bind', '/usr', '/usr', '--ro-bind', '/lib', '/lib',
           '--ro-bind', '/lib64', '/lib64', '--proc', '/proc', '--dev', '/dev',
           '--tmpfs', '/tmp', '--chdir', '/tmp', '/usr/bin/python3', '-I', '-']
    try:
        with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
            r = subprocess.run(cmd, input=program.encode(), stdout=out, stderr=err,
                               timeout=6, preexec_fn=limit)
            out.seek(max(0, out.tell() - 16384))
            stdout = out.read().decode('utf-8', errors='replace')
            err.seek(max(0, err.tell() - 1000))
            stderr = err.read().decode('utf-8', errors='replace')
    except subprocess.TimeoutExpired:
        return {'error': 'timeout'}
    # Output files are resource-limited; inspect only a bounded tail.
    lines = stdout.splitlines()
    for line in reversed(lines):
        if line.startswith('__CODING_RESULT__'):
            try:
                result = json.loads(line[len('__CODING_RESULT__'):])
            except ValueError:
                break
            if 'cases' in result:
                result['passed'] = sum(x is True for x in result['cases'])
                result['total'] = len(cfg['tests']) + (name == 'merge_intervals')
                result['all_passed'] = result['passed'] == result['total']
            return result
    return {'error': 'sandbox exited without result', 'exit_code': r.returncode,
            'stderr_tail': stderr}


if __name__ == '__main__':
    with open(sys.argv[1]) as f:
        raw = json.load(f)
    result = {'model': raw['model'], 'mode': raw['mode'], 'engine_sha256': raw['engine_sha256'], 'cases': []}
    for item in raw['cases']:
        result['cases'].append({'name': item['name'], 'evaluation': evaluate(item['name'], item['answer'])})
    print(json.dumps(result, indent=2))
