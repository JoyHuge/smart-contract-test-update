import difflib
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, Any, List

from codebleu import calc_codebleu


class TestComparator:
    """Compare LLM-generated tests with developer-written ground-truth tests."""

    # ------------------------------------------------------------------
    
    # ------------------------------------------------------------------

    def extract_test_names_via_ast(self, test_source: str) -> List[str]:
        """Extract string arguments from test declarations with Acorn."""
        script_path = Path(__file__).parent / "extract_test_names.js"
        if not script_path.exists():
            print("  ⚠️  extract_test_names.js not found, falling back to regex")
            return self._extract_test_functions_regex(test_source)

        try:
            result = subprocess.run(
                ["node", str(script_path)],
                input=test_source,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                names = json.loads(result.stdout.strip())
                
                return [n for n in names if n and (not n[0].isupper() or ' ' in n)]
            print("  ⚠️  Acorn parse failed, falling back to regex")
            return self._extract_test_functions_regex(test_source)
        except Exception as e:
            print(f"  ⚠️  AST parse failed: {e}, falling back to regex")
            return self._extract_test_functions_regex(test_source)

    @staticmethod
    def _extract_test_functions_regex(test: str) -> List[str]:
        """Extract test names with a regular-expression fallback."""
        pattern = r'(?:it|describe)\(\s*[`"\']([^"`\']+?)[`"\']'
        return re.findall(pattern, test)

    # ------------------------------------------------------------------
    
    # ------------------------------------------------------------------

    @staticmethod
    def parse_passing_tests_from_log(log: str) -> List[str]:
        """Extract passing test names from test-run output.

        Supported Mocha, Truffle, and Hardhat formats include:
          "    ✓ test name (49ms)"
          "    ✓ test name"
          "    ✔ test name"
        The Foundry format is also supported:
          "[PASS] test_name (gas: X)"
        """
        if not log:
            return []

        
        clean_log = re.sub(r'\x1b\[[0-9;]*m', '', log)

        names = []

        
        for m in re.finditer(r'^\s*[✓✔]\s+(.+?)(?:\s*\(\d+m?s\))?\s*$', clean_log, re.MULTILINE):
            name = m.group(1).strip()
            if name:
                names.append(name)

        # Foundry: "[PASS] test_name (gas: X)"
        if not names:
            for m in re.finditer(r'^\s*\[PASS\]\s+(\S+)', clean_log, re.MULTILINE):
                names.append(m.group(1))

        return names

    # ------------------------------------------------------------------
    
    # ------------------------------------------------------------------

    @staticmethod
    def parse_test_run_result(log: str, framework: str = "truffle") -> Dict[str, Any]:
        """Parse compilation and test results from a test-run log.

        Returns:
            compilation: 'pass' | 'fail'
            test_result: 'pass' | 'fail' | 'N/A' (N/A after compilation failure)
            passing_count: int
            failing_count: int
        """
        if not log:
            return {
                'compilation': 'fail',
                'test_result': 'N/A',
                'passing_count': 0,
                'failing_count': 0,
            }

        clean_log = re.sub(r'\x1b\[[0-9;]*m', '', log)
        fw = framework.strip().lower()

        compilation = 'fail'
        test_result = 'N/A'
        passing_count = 0
        failing_count = 0

        if fw == "truffle":
            
            if re.search(r'Compiled successfully', clean_log) or \
               re.search(r'nothing to compile', clean_log, re.IGNORECASE):
                compilation = 'pass'
            
            m_pass = re.search(r'(\d+)\s+passing', clean_log)
            m_fail = re.search(r'(\d+)\s+failing', clean_log)
            if m_pass:
                passing_count = int(m_pass.group(1))
                compilation = 'pass'
                test_result = 'pass' if (not m_fail or int(m_fail.group(1)) == 0) else 'fail'
                if m_fail:
                    failing_count = int(m_fail.group(1))

        elif fw == "hardhat":
            
            if re.search(r'Compiled\s+\d+\s+Solidity file', clean_log) or \
               re.search(r'compilation finished successfully', clean_log, re.IGNORECASE) or \
               re.search(r'nothing to compile', clean_log, re.IGNORECASE):
                compilation = 'pass'
            m_pass = re.search(r'(\d+)\s+passing', clean_log)
            m_fail = re.search(r'(\d+)\s+failing', clean_log)
            if m_pass:
                passing_count = int(m_pass.group(1))
                compilation = 'pass'
                test_result = 'pass' if (not m_fail or int(m_fail.group(1)) == 0) else 'fail'
                if m_fail:
                    failing_count = int(m_fail.group(1))

        elif fw == "foundry":
            
            if re.search(r'Suite result:', clean_log):
                compilation = 'pass'
            m_pass = re.search(r'(\d+)\s+passed', clean_log)
            m_fail = re.search(r'(\d+)\s+failed', clean_log)
            if m_pass:
                passing_count = int(m_pass.group(1))
                failing_count = int(m_fail.group(1)) if m_fail else 0
                test_result = 'pass' if failing_count == 0 else 'fail'

        return {
            'compilation': compilation,
            'test_result': test_result,
            'passing_count': passing_count,
            'failing_count': failing_count,
        }

    # ------------------------------------------------------------------
    
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_test_name(name: str) -> str:
        """Normalize a test title for conservative name matching."""
        normalized = (name or "").lower()
        normalized = normalized.replace("-", " ")
        normalized = re.sub(r"[^a-z0-9_]+", " ", normalized)
        words = []
        weak_words = {"should", "when", "that", "the", "a", "an", "with", "to"}
        stems = {
            "removing": "remove",
            "removed": "remove",
            "removes": "remove",
            "trying": "try",
            "indices": "index",
        }
        for word in normalized.split():
            word = stems.get(word, word)
            if word.endswith("ing") and len(word) > 5:
                word = word[:-3]
            if word.endswith("ed") and len(word) > 4:
                word = word[:-2]
            if word.endswith("s") and len(word) > 4:
                word = word[:-1]
            if word not in weak_words:
                words.append(word)
        return " ".join(words)

    @staticmethod
    def compare_test_case_names(
        llm_cases: List[str], ground_truth_cases: List[str]
    ) -> Dict[str, Any]:
        """Compare test-case names from two versions, including order."""
        if not ground_truth_cases:
            ucr = "N/A"
        else:
            matched = len(set(llm_cases) & set(ground_truth_cases))
            ucr = round(matched / len(ground_truth_cases) * 100, 2)

        return {
            'ucr': ucr,
            'llm_passing_count': len(llm_cases),
            'ground_truth_passing_count': len(ground_truth_cases),
            'llm_only': [c for c in llm_cases if c not in ground_truth_cases],
            'ground_truth_only': [
                c for c in ground_truth_cases if c not in llm_cases
            ],
        }

    def compare_test_case_semantics(
        self,
        llm_cases: List[str],
        ground_truth_cases: List[str],
        llm_test_source: str,
        ground_truth_test_source: str,
    ) -> Dict[str, Any]:
        """Compare passing test cases using exact, normalized, then semantic matching."""
        if not ground_truth_cases:
            return {
                'ucr': "N/A",
                'llm_passing_count': len(llm_cases),
                'ground_truth_passing_count': 0,
                'llm_only': llm_cases,
                'ground_truth_only': [],
                'matches': [],
                'match_summary': {'exact': 0, 'normalized': 0, 'semantic': 0},
            }

        llm_signatures = self.extract_test_case_signatures(llm_test_source)
        ground_truth_signatures = self.extract_test_case_signatures(
            ground_truth_test_source
        )

        used_llm: set[int] = set()
        matches: List[Dict[str, Any]] = []
        match_summary = {'exact': 0, 'normalized': 0, 'semantic': 0}

        for ground_truth_name in ground_truth_cases:
            ground_truth_sig = ground_truth_signatures.get(
                ground_truth_name,
                self.empty_case_signature(ground_truth_name),
            )
            best = None
            for idx, llm_name in enumerate(llm_cases):
                if idx in used_llm:
                    continue
                llm_sig = llm_signatures.get(llm_name, self.empty_case_signature(llm_name))
                match = self.match_case_signature(llm_sig, ground_truth_sig)
                if match:
                    best = (idx, match)
                    if match['match_type'] == 'exact':
                        break
            if best:
                idx, match = best
                used_llm.add(idx)
                match_summary[match['match_type']] += 1
                matches.append({
                    'ground_truth': ground_truth_name,
                    'llm': llm_cases[idx],
                    **match,
                })

        matched = len(matches)
        ucr = round(matched / len(ground_truth_cases) * 100, 2)
        matched_ground_truth = {m['ground_truth'] for m in matches}
        matched_llm = {m['llm'] for m in matches}

        return {
            'ucr': ucr,
            'llm_passing_count': len(llm_cases),
            'ground_truth_passing_count': len(ground_truth_cases),
            'llm_only': [c for c in llm_cases if c not in matched_llm],
            'ground_truth_only': [
                c for c in ground_truth_cases if c not in matched_ground_truth
            ],
            'matches': matches,
            'match_summary': match_summary,
        }

    def match_case_signature(
        self, llm_sig: Dict[str, Any], ground_truth_sig: Dict[str, Any]
    ) -> Dict[str, Any] | None:
        if llm_sig['name'] == ground_truth_sig['name']:
            return {'match_type': 'exact', 'evidence': {'name': llm_sig['name']}}

        if (
            llm_sig['normalized_name']
            and llm_sig['normalized_name'] == ground_truth_sig['normalized_name']
        ):
            return {
                'match_type': 'normalized',
                'evidence': {'normalized_name': llm_sig['normalized_name']},
            }

        semantic_evidence = self.semantic_match_evidence(llm_sig, ground_truth_sig)
        if semantic_evidence:
            return {'match_type': 'semantic', 'evidence': semantic_evidence}
        return None

    @staticmethod
    def semantic_match_evidence(
        llm_sig: Dict[str, Any], ground_truth_sig: Dict[str, Any]
    ) -> Dict[str, Any] | None:
        llm_functions = set(llm_sig['called_functions'])
        ground_truth_functions = set(ground_truth_sig['called_functions'])
        common_functions = sorted(llm_functions & ground_truth_functions)
        llm_assertions = set(llm_sig['assertion_kinds'])
        ground_truth_assertions = set(ground_truth_sig['assertion_kinds'])
        common_assertions = sorted(llm_assertions & ground_truth_assertions)

        if 'revert' in llm_assertions and 'revert' in ground_truth_assertions:
            llm_reasons = set(llm_sig['normalized_revert_reasons'])
            ground_truth_reasons = set(
                ground_truth_sig['normalized_revert_reasons']
            )
            common_reasons = sorted(llm_reasons & ground_truth_reasons)
            if common_functions and common_reasons:
                return {
                    'kind': 'revert',
                    'called_functions': common_functions,
                    'revert_reasons': common_reasons,
                }

        llm_events = set(llm_sig['event_names'])
        ground_truth_events = set(ground_truth_sig['event_names'])
        common_events = sorted(llm_events & ground_truth_events)
        if common_functions and common_events:
            return {
                'kind': 'event',
                'called_functions': common_functions,
                'event_names': common_events,
            }

        if common_functions and common_assertions:
            llm_tokens = set(llm_sig['normalized_name'].split())
            ground_truth_tokens = set(ground_truth_sig['normalized_name'].split())
            shared_tokens = sorted(llm_tokens & ground_truth_tokens)
            if len(shared_tokens) >= 2:
                return {
                    'kind': 'assertion',
                    'called_functions': common_functions,
                    'assertion_kinds': common_assertions,
                    'shared_name_tokens': shared_tokens,
                }

        return None

    def extract_test_case_signatures(self, test_source: str) -> Dict[str, Dict[str, Any]]:
        signatures: Dict[str, Dict[str, Any]] = {}
        for case in self.extract_test_cases(test_source):
            signatures[case['name']] = self.build_case_signature(case['name'], case['body'])
        return signatures

    def extract_test_cases(self, test_source: str) -> List[Dict[str, str]]:
        """Extract it(...) blocks with simple delimiter matching."""
        cases: List[Dict[str, str]] = []
        pattern = re.compile(r"\bit\s*\(\s*(['\"`])((?:\\.|(?!\1).)*?)\1\s*,", re.DOTALL)
        for match in pattern.finditer(test_source or ""):
            name = match.group(2).strip()
            block_start = match.start()
            open_paren = test_source.find("(", match.start())
            block_end = self.find_matching_delimiter(test_source, open_paren, "(", ")")
            if block_end == -1:
                body = test_source[match.end():]
            else:
                body = test_source[match.end():block_end]
            cases.append({'name': name, 'body': body})
        return cases

    @staticmethod
    def find_matching_delimiter(text: str, start: int, open_char: str, close_char: str) -> int:
        if start < 0 or start >= len(text) or text[start] != open_char:
            return -1
        depth = 0
        quote = None
        escaped = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if quote:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == quote:
                    quote = None
                continue
            if ch in {"'", '"', "`"}:
                quote = ch
                continue
            if ch == open_char:
                depth += 1
            elif ch == close_char:
                depth -= 1
                if depth == 0:
                    return idx
        return -1

    def build_case_signature(self, name: str, body: str) -> Dict[str, Any]:
        reasons = self.extract_revert_reasons(body)
        return {
            'name': name,
            'normalized_name': self.normalize_test_name(name),
            'called_functions': self.extract_called_functions(body),
            'assertion_kinds': self.extract_assertion_kinds(body),
            'revert_reasons': reasons,
            'normalized_revert_reasons': [self.normalize_literal(reason) for reason in reasons],
            'event_names': self.extract_event_names(body),
        }

    def empty_case_signature(self, name: str) -> Dict[str, Any]:
        return {
            'name': name,
            'normalized_name': self.normalize_test_name(name),
            'called_functions': [],
            'assertion_kinds': [],
            'revert_reasons': [],
            'normalized_revert_reasons': [],
            'event_names': [],
        }

    @staticmethod
    def normalize_literal(value: str) -> str:
        normalized = re.sub(r"\s+", " ", (value or "").strip().lower())
        normalized = re.sub(r"^(?:vm exception while processing transaction:\s*)?revert\s+", "", normalized)
        return normalized

    @staticmethod
    def extract_called_functions(body: str) -> List[str]:
        names = []
        ignored = {
            'require', 'assert', 'equal', 'include', 'reverts', 'eventEmitted',
            'inLogs', 'emit', 'expect', 'to', 'be', 'revertedWith', 'new',
            'fail',
        }
        for name in re.findall(r"\.([A-Za-z_]\w*)\s*\(", body or ""):
            if name not in ignored and name not in names:
                names.append(name)
        return names

    @staticmethod
    def extract_assertion_kinds(body: str) -> List[str]:
        kinds = []
        patterns = [
            ('revert', r"\b(?:reverts|expectRevert|revertedWith|reverted)\b"),
            ('revert', r"\berror\.message\b"),
            ('revert', r"\bExpected revert\b"),
            ('event', r"\b(?:expectEvent|eventEmitted|emit)\b"),
            ('equal', r"\b(?:assert\.equal|assert\.strictEqual|expect\([^)]*\)\.to\.equal)\b"),
            ('include', r"\b(?:include|contains)\b"),
        ]
        for kind, pattern in patterns:
            if re.search(pattern, body or "") and kind not in kinds:
                kinds.append(kind)
        return kinds

    @staticmethod
    def extract_revert_reasons(body: str) -> List[str]:
        reasons = []
        for value in re.findall(r"['\"]([^'\"]+)['\"]", body or ""):
            normalized = TestComparator.normalize_literal(value)
            if not normalized:
                continue
            if normalized in {"expected revert not received", "expected revert"}:
                continue
            looks_like_revert_reason = (
                "revert" in value.lower()
                or "error.message" in (body or "")
                or any(token in normalized for token in [
                    "out of range", "invalid", "not found", "not allowed",
                    "only", "exceed", "insufficient", "unauthorized",
                    "forbidden", "zero address",
                ])
            )
            if looks_like_revert_reason and value not in reasons:
                reasons.append(value)
        return reasons

    @staticmethod
    def extract_event_names(body: str) -> List[str]:
        events = []
        for pattern in [
            r"\b(?:expectEvent|eventEmitted)\s*\((?P<args>.*?)\)",
            r"\.to\.emit\s*\([^)]*\)\.withArgs",
            r"\.to\.emit\s*\([^,]+,\s*['\"](?P<event>[^'\"]+)['\"]",
        ]:
            for match in re.finditer(pattern, body or "", re.DOTALL):
                if "event" in match.groupdict() and match.group("event"):
                    values = [match.group("event")]
                else:
                    values = re.findall(r"['\"]([^'\"]+)['\"]", match.group("args"))
                for value in values:
                    if value and value not in events:
                        events.append(value)
        return events

    # ------------------------------------------------------------------
    
    # ------------------------------------------------------------------

    def compare_tests(
        self,
        llm_test: str,
        ground_truth_test: str
    ) -> Dict[str, Any]:
        """Compare two test files."""
        comparison = {
            'line_count_diff': len(llm_test.split('\n')) - len(ground_truth_test.split('\n')),
            'char_count_diff': len(llm_test) - len(ground_truth_test),
            'similarity': self.calculate_codebleu(llm_test, ground_truth_test),
            'diff': self.generate_diff(llm_test, ground_truth_test),
            'function_coverage': self.compare_function_coverage(
                llm_test, ground_truth_test
            )
        }
        return comparison

    def calculate_codebleu(self, text1: str, text2: str) -> float:
        """Calculate a CodeBLEU score between 0 and 1."""
        try:
            result = calc_codebleu(
                references=[text2],
                predictions=[text1],
                lang="javascript",
            )
            return result.get('codebleu', 0.0) if isinstance(result, dict) else float(result)
        except Exception as e:
            print(f"  ❌ CodeBLEU calculation failed: {type(e).__name__}: {e}")
            return 0.0

    @staticmethod
    def check_syntax(file_path: str) -> bool:
        """Validate JavaScript syntax with ``node --check``."""
        try:
            result = subprocess.run(
                ["node", "--check", file_path],
                capture_output=True, text=True, timeout=10,
            )
            return result.returncode == 0
        except Exception:
            return False

    def generate_diff(self, text1: str, text2: str) -> str:
        """Generate a unified diff."""
        diff = difflib.unified_diff(
            text1.splitlines(keepends=True),
            text2.splitlines(keepends=True),
            fromfile='LLM_Test',
            tofile='Ground_Truth_Test',
            lineterm=''
        )
        return ''.join(diff)

    def compare_function_coverage(
        self,
        test1: str,
        test2: str
    ) -> Dict[str, Any]:
        """Compare test-function coverage by exact and semantic matches."""
        funcs1 = self.extract_test_names_via_ast(test1)
        funcs2 = self.extract_test_names_via_ast(test2)
        semantic = self.compare_test_case_semantics(funcs1, funcs2, test1, test2)

        return {
            'llm_test_functions': len(funcs1),
            'ground_truth_test_functions': len(funcs2),
            'common_functions': len(semantic.get('matches', [])),
            'llm_only': semantic.get('llm_only', []),
            'ground_truth_only': semantic.get('ground_truth_only', []),
            'matches': semantic.get('matches', []),
            'match_summary': semantic.get('match_summary', {}),
        }

    def generate_evaluation_report(
        self,
        comparison: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> str:
        """Generate an evaluation report."""
        report = []
        report.append("=" * 60)
        report.append("TEST GENERATION EVALUATION REPORT")
        report.append("=" * 60)

        report.append(f"\nRecord Information:")
        report.append(f"  Repo: {metadata.get('repo_name', 'N/A')}")
        report.append(f"  Contract: {metadata.get('production_file_path', 'N/A')}")
        report.append(f"  Test: {metadata.get('test_file_path', 'N/A')}")

        report.append(f"\nComparison Metrics:")
        report.append(f"  CodeBLEU Score: {comparison['similarity']:.4f}")
        report.append(f"  Line Count Diff: {comparison['line_count_diff']}")
        report.append(f"  Char Count Diff: {comparison['char_count_diff']}")

        coverage = comparison['function_coverage']
        report.append(f"\nTest Function Coverage:")
        report.append(f"  LLM Test Functions: {coverage['llm_test_functions']}")
        report.append(
            f"  Ground Truth Test Functions: "
            f"{coverage['ground_truth_test_functions']}"
        )
        report.append(f"  Common Functions: {coverage['common_functions']}")

        if coverage['llm_only']:
            report.append(f"  LLM Only Functions: {coverage['llm_only']}")
        if coverage['ground_truth_only']:
            report.append(
                f"  Ground Truth Only Functions: {coverage['ground_truth_only']}"
            )

        report.append(f"\nDiff Preview (first 30 lines):")
        report.append("-" * 60)
        report.append(comparison['diff'][:2000])

        return '\n'.join(report)
