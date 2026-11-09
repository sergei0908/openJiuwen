"""Запуск тестов задачи coding, когда нельзя просто `python -m unittest`: тесты в подпапке или в стиле
pytest (функции test_* с assert), а самого pytest в контейнере нет.

    python harness/testrun.py tests/test_mylib.py [...]      # из папки задачи

Собирает unittest.TestCase и функции test_* без аргументов, печатает вывод в формате unittest
(«FAIL: test_x (module)», «Ran N tests», «FAILED (...)»: по нему работают сжатие вывода и выбор
упавших функций для агента). Код выхода 1, если что-то упало или не нашлось ни одного теста:
«0 тестов прошло» - это не «всё исправлено».
"""
import importlib.util
import inspect
import os
import sys
import unittest


class FunctionCase(unittest.FunctionTestCase):
    """Функция test_* как тест; в выводе - имя функции, как у методов TestCase."""

    def id(self):
        return f"{self._testFunc.__module__}.{self._testFunc.__name__}"

    def __str__(self):
        return f"{self._testFunc.__name__} ({self._testFunc.__module__})"


def load_suite(module):
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    for name, obj in vars(module).items():
        if name.startswith("test") and inspect.isfunction(obj) and obj.__module__ == module.__name__ \
                and not inspect.signature(obj).parameters:
            suite.addTest(FunctionCase(obj))
    return suite


def main(paths):
    root = os.getcwd()
    sys.path.insert(0, root)
    suite = unittest.TestSuite()
    for p in paths:
        p = os.path.abspath(p)
        d = os.path.dirname(p)
        if d not in sys.path:
            sys.path.insert(0, d)
        name = os.path.splitext(os.path.basename(p))[0]
        spec = importlib.util.spec_from_file_location(name, p)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        suite.addTests(load_suite(mod))
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=1).run(suite)
    return 0 if result.wasSuccessful() and result.testsRun > 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
