"""Pure-Python tests for sparkx_si.verifiers incl. adversarial cases (positive / negative / hack attempts)."""
import json
import os
import socket
import sys
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sparkx_si import verifiers as V


class TestThinkAndMath(unittest.TestCase):
    def test_answer_part(self):
        self.assertEqual(V.answer_part("no tags"), "no tags")
        self.assertEqual(V.answer_part("<think>a</think>b").strip(), "b")
        self.assertIsNone(V.answer_part("<think>never closed \\boxed{4}"))
        self.assertIsNone(V.answer_part("<think>a</think>b<think>c"))

    def test_boxed_in_think_does_not_count(self):
        self.assertFalse(V.verify_math("<think>so \\boxed{42}</think>I am not sure.", "42").ok)
        self.assertTrue(V.verify_math("<think>x</think>The result is \\boxed{42}.", "42").ok)

    def test_unclosed_think_fails(self):
        v = V.verify_math("<think>... \\boxed{42}", "42")
        self.assertFalse(v.ok)
        self.assertEqual(v.reason, "no_answer_after_think")

    def test_nested_boxed_and_last_wins(self):
        self.assertEqual(V.extract_boxed(r"first \boxed{1} then \boxed{\frac{3}{4}}"), r"\frac{3}{4}")

    def test_numeric_normalisation(self):
        eq = V.math_equal
        self.assertTrue(eq("1,234", "1234"))
        self.assertTrue(eq("0.5", r"\frac{1}{2}"))
        self.assertTrue(eq(r"\dfrac{3}{6}", "1/2"))
        self.assertTrue(eq("50%", "0.5"))
        self.assertTrue(eq("$12.00$", "12"))
        self.assertFalse(eq(r"12\text{ cm}", "12"))  # units are NOT silently stripped (stricter = safer reward)
        self.assertFalse(eq("41", "42"))
        self.assertFalse(eq("", "42"))
        self.assertFalse(eq(None, "42"))
        self.assertTrue(eq("x+1", "X + 1"))
        self.assertFalse(eq("x+1", "x+2"))

    def test_fallback_extractors(self):
        self.assertTrue(V.verify_math("work...\n#### 18", "18").ok)
        self.assertTrue(V.verify_math("So the answer is 18.", "18").ok)
        self.assertTrue(V.verify_math("it is 5 then 18", "18").ok)  # last number

    def test_answer_only_in_prompt_text_is_not_credited(self):
        # response that merely repeats the question number does not match a different gold
        self.assertFalse(V.verify_math("The question mentions 7 apples.", "42").ok)


class TestQA(unittest.TestCase):
    def test_alias_and_normalisation(self):
        self.assertTrue(V.verify_qa("The Eiffel Tower", ["eiffel tower"]).ok)
        self.assertTrue(V.verify_qa("<think>hmm</think>Answer: Paris.", ["Paris"]).ok)

    def test_hedge_dump_fails(self):
        dump = "It could be Paris, Lyon, Marseille, Toulouse, Nice, Nantes, Strasbourg, Montpellier, Bordeaux, Lille or Rennes."
        self.assertFalse(V.verify_qa(dump, ["Paris"]).ok)

    def test_wrong_and_empty(self):
        self.assertFalse(V.verify_qa("London", ["Paris"]).ok)
        self.assertFalse(V.verify_qa("<think>x</think>", ["Paris"]).ok)

    def test_think_only_answer_fails(self):
        self.assertFalse(V.verify_qa("<think>Paris</think>I don't know", ["Paris"]).ok)


class TestIF(unittest.TestCase):
    def c(self, i, **a):
        return {"id": i, "args": a}

    def test_each_constraint_pos_neg(self):
        T = V.check_constraint
        self.assertTrue(T(self.c("min_words", n=3), "one two three four"))
        self.assertFalse(T(self.c("min_words", n=5), "one two three four"))
        self.assertTrue(T(self.c("max_words", n=4), "one two three four"))
        self.assertFalse(T(self.c("max_words", n=3), "one two three four"))
        self.assertTrue(T(self.c("max_sentences", n=2), "A b. C d."))
        self.assertFalse(T(self.c("max_sentences", n=1), "A b. C d."))
        self.assertTrue(T(self.c("exact_bullets", n=2), "- a\n- b"))
        self.assertFalse(T(self.c("exact_bullets", n=2), "- a\n- b\n- c"))
        self.assertTrue(T(self.c("exact_paragraphs", n=2), "p1\n\np2"))
        self.assertFalse(T(self.c("exact_paragraphs", n=2), "p1 p2"))
        self.assertTrue(T(self.c("forbidden_words", words=["bad"]), "good stuff"))
        self.assertFalse(T(self.c("forbidden_words", words=["bad"]), "this is BAD"))
        self.assertTrue(T(self.c("required_words", words=["river"]), "A River runs"))
        self.assertFalse(T(self.c("required_words", words=["river"]), "riverbank"))  # whole word
        self.assertTrue(T(self.c("keyword_min_count", word="x", n=2), "x and x"))
        self.assertTrue(T(self.c("all_lowercase"), "abc def"))
        self.assertFalse(T(self.c("all_lowercase"), "Abc"))
        self.assertFalse(T(self.c("all_uppercase"), "12345"))  # needs letters
        self.assertTrue(T(self.c("no_commas"), "no commas here"))
        self.assertFalse(T(self.c("no_commas"), "a, b"))
        self.assertTrue(T(self.c("ends_with", phrase="Bye."), "Hello. Bye.\n"))
        self.assertTrue(T(self.c("title_in_brackets"), "<<My Title>>\nbody"))
        self.assertFalse(T(self.c("title_in_brackets"), "<<>>"))
        self.assertTrue(T(self.c("json_format"), '```json\n{"a": 1}\n```'))
        self.assertFalse(T(self.c("json_format"), "not json"))

    def test_prompt_level_and_loose(self):
        cons = [self.c("all_lowercase"), self.c("no_commas")]
        self.assertTrue(V.verify_if("hello world", cons).ok)
        self.assertFalse(V.verify_if("Hello world", cons).ok)
        # loose: ignores a markdown-bold marker and a polite first line
        loose = V.verify_if("Sure! Here you go\nhello world", cons, strict=False)
        self.assertTrue(loose.ok)
        self.assertFalse(V.verify_if("Sure! Here you go\nhello world", cons, strict=True).ok)

    def test_constraint_in_think_only_fails(self):
        self.assertFalse(V.verify_if("<think>all lowercase</think>Hello", [self.c("all_lowercase")]).ok)

    def test_empty_constraints_do_not_pass(self):
        self.assertFalse(V.verify_if("anything", []).ok)


class TestTool(unittest.TestCase):
    tools = {"add": {"type": "object", "required": ["a", "b"], "additionalProperties": False,
                     "properties": {"a": {"type": "integer"}, "b": {"type": "integer", "minimum": 0}}}}
    impl = {"add": lambda a, b: a + b}
    exp = {"name": "add", "result": 5}

    def call(self, obj):
        return V.verify_tool_call("<think>x</think>```json\n" + json.dumps(obj) + "\n```", self.tools, self.exp, self.impl)

    def test_ok(self):
        self.assertTrue(self.call({"name": "add", "arguments": {"a": 2, "b": 3}}).ok)

    def test_failures(self):
        self.assertEqual(self.call({"name": "add", "arguments": {"a": 2}}).reason, "schema_violation")
        self.assertEqual(self.call({"name": "add", "arguments": {"a": "2", "b": 3}}).reason, "schema_violation")
        self.assertEqual(self.call({"name": "add", "arguments": {"a": 2, "b": -3}}).reason, "schema_violation")
        self.assertEqual(self.call({"name": "add", "arguments": {"a": 2, "b": 3, "c": 1}}).reason, "schema_violation")
        self.assertEqual(self.call({"name": "mul", "arguments": {}}).reason, "unknown_tool")
        self.assertEqual(self.call({"name": "add", "arguments": {"a": 2, "b": 4}}).reason, "wrong_result")
        self.assertFalse(V.verify_tool_call("no json here", self.tools, self.exp, self.impl).ok)

    def test_bool_is_not_integer(self):
        self.assertEqual(self.call({"name": "add", "arguments": {"a": True, "b": 3}}).reason, "schema_violation")

    def test_tool_call_tags(self):
        r = '<tool_call>{"name": "add", "arguments": {"a": 1, "b": 4}}</tool_call>'
        self.assertTrue(V.verify_tool_call(r, self.tools, self.exp, self.impl).ok)


class TestSandbox(unittest.TestCase):
    TESTS = "assert add(2, 3) == 5\nassert add(-1, 1) == 0\nassert add(10, 20) == 30\n"

    def test_correct_code_passes(self):
        self.assertTrue(V.run_python_tests("def add(a, b):\n    return a + b\n", self.TESTS).ok)

    def test_wrong_code_fails(self):
        self.assertFalse(V.run_python_tests("def add(a, b):\n    return a - b\n", self.TESTS).ok)

    def test_printing_expected_output_fails(self):
        v = V.run_python_tests("print(5)\n", self.TESTS)
        self.assertFalse(v.ok)

    def test_sys_exit_and_os_exit_hijack_fail(self):
        self.assertFalse(V.run_python_tests("import sys\nsys.exit(0)\n", self.TESTS).ok)
        self.assertFalse(V.run_python_tests("import os\nos._exit(0)\n", self.TESTS).ok)

    def test_printing_sentinel_without_nonce_fails(self):
        self.assertFalse(V.run_python_tests("print('__SI_OK__')\nimport os\nos._exit(0)\n", self.TESTS).ok)

    def test_timeout(self):
        v = V.run_python_tests("while True:\n    pass\n", self.TESTS, timeout=1.5)
        self.assertFalse(v.ok)
        self.assertIn(v.reason, ("timeout", "tests_failed"))

    def test_env_is_scrubbed(self):
        os.environ["HF_TOKEN_SI_TEST"] = "secret"
        try:
            code = "import os\nassert 'HF_TOKEN_SI_TEST' not in os.environ\ndef add(a, b):\n    return a + b\n"
            self.assertTrue(V.run_python_tests(code, self.TESTS).ok)
        finally:
            del os.environ["HF_TOKEN_SI_TEST"]

    def test_network_blocked(self):
        srv = socket.socket()
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        srv.settimeout(0.5)
        port = srv.getsockname()[1]
        hit = []

        def accept():
            try:
                c, _ = srv.accept()
                hit.append(1)
                c.close()
            except Exception:
                pass
        th = threading.Thread(target=accept)
        th.start()
        code = (f"import socket\ns = socket.socket()\ns.connect(('127.0.0.1', {port}))\n"
                "def add(a, b):\n    return a + b\n")
        v = V.run_python_tests(code, self.TESTS, timeout=3)
        th.join()
        srv.close()
        self.assertFalse(v.ok)
        self.assertEqual(hit, [])

    def test_verify_code_extracts_last_block(self):
        resp = "<think>t</think>```python\ndef add(a,b):\n    return 0\n```\nactually:\n```python\ndef add(a, b):\n    return a + b\n```"
        self.assertTrue(V.verify_code(resp, self.TESTS).ok)
        self.assertEqual(V.verify_code("no code", self.TESTS).reason, "no_code_block")


class TestDispatch(unittest.TestCase):
    def test_dispatch(self):
        t = {"pillar": "math", "spec": {"gold": "4"}}
        self.assertTrue(V.verify(t, "\\boxed{4}").ok)
        with self.assertRaises(KeyError):
            V.verify({"pillar": "nope", "spec": {}}, "x")


if __name__ == "__main__":
    unittest.main()
