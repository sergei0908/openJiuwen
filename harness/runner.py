"""Планировщик: бесплатные решения -> отбор задач по ценности/стоимости -> LLM-решение."""
import json
import os
import random
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import config, oj_agent, solvers
from .estimator import Estimator
from .llm import LLM, BudgetExhausted, Ledger
from .oj_agent import TaskBudgetExceeded
from .tasks import AnswerWriter, can_write, find_budget, load_tasks


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class TaskLLM:
    """Бюджет одной задачи. LLM-решение идёт через агента openJiuwen (harness/oj_agent.py).

    Каждый вызов модели агентом списывается и с общего бюджета, и с бюджета задачи; при
    перерасходе задачи агент обрывается (TaskBudgetExceeded).
    """

    def __init__(self, llm, cap):
        self.llm, self.cap, self.spent, self.calls = llm, cap, 0.0, []
        self.ledger = llm.ledger
        self._lock = threading.Lock()

    def record(self, model, prompt_tokens, completion_tokens, cost):
        with self._lock:
            self.spent += cost
            self.calls.append({"model": model, "prompt_tokens": prompt_tokens,
                               "completion_tokens": completion_tokens, "cost": round(cost, 1)})

    def agent(self, *, system_prompt, query, tools, model=None, max_iterations=4, max_tokens=800,
              stop=None):
        """Запускает агента openJiuwen и возвращает его финальный текст."""
        return oj_agent.run_agent(self.llm, self, system_prompt=system_prompt, query=query,
                                  tools=tools, model_name=model or config.MODEL_FAST,
                                  max_iterations=max_iterations, max_tokens=max_tokens, stop=stop)

    def chat(self, messages, model=None, max_tokens=None):
        if self.spent >= self.cap:
            raise TaskBudgetExceeded(f"task cap {self.cap:.0f}")
        rem = self.ledger.remaining()
        if rem != float("inf") and max_tokens:
            # не заказываем больше выходных токенов, чем осталось в общем бюджете
            max_tokens = int(max(50, min(max_tokens, rem)))
        text, cost, meta = self.llm.chat(messages, model=model, max_tokens=max_tokens)
        self.spent += cost
        self.calls.append(dict(meta, cost=round(cost, 1)))
        return text, cost, meta


class Runner:
    def __init__(self, tasks_dir, out=None):
        self.tasks_dir = Path(tasks_dir)
        self.t0 = time.time()
        self.tasks = load_tasks(self.tasks_dir)
        self.ledger = Ledger(find_budget(self.tasks_dir))
        self.llm = LLM(self.ledger)
        self.est = Estimator()
        self.writer = AnswerWriter(out)
        self.traj_lock = threading.Lock()
        self.stats = {"fast": 0, "llm_ok": 0, "llm_fail": 0, "skipped_infeasible": 0,
                      "skipped_budget": 0, "reward_claimed": 0.0, "answers_lost": 0}

    def time_left(self):
        return config.TIME_LIMIT_S - (time.time() - self.t0)

    def save(self, task, answer):
        """Баллы засчитываем, только если ответ действительно сохранён."""
        if self.writer.write(task, answer):
            self.stats["reward_claimed"] += task.reward
        else:
            self.stats["answers_lost"] += 1

    def traj(self, rec):
        try:
            with self.traj_lock, open(config.TRAJ_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError:
            pass

    # ---------- фаза A: всё, что решается без LLM ----------
    def analyze_one(self, task):
        mod = solvers.pick(task)
        t = time.time()
        try:
            plan = mod.analyze(task, fast_path=config.FAST_PATH)
        except Exception as e:  # noqa: BLE001
            log("analyze error", task.task_id, e)
            plan = solvers.generic.analyze(task)
        plan.notes["analyze_s"] = round(time.time() - t, 2)
        if plan.fast_answer is not None and plan.feasible is not False:
            self.save(task, plan.fast_answer)
            self.stats["fast"] += 1
            self.traj({"task_id": task.task_id, "type": task.task_type, "reward": task.reward,
                       "route": "fast", "cost": 0, "answer": plan.fast_answer})
        elif plan.feasible is False:
            self.stats["skipped_infeasible"] += 1
            self.traj({"task_id": task.task_id, "type": task.task_type, "reward": task.reward,
                       "route": "infeasible", "cost": 0})
        return task, mod, plan

    # ---------- фаза B: LLM по приоритету ----------
    def solve_one(self, task, mod, plan, pred):
        cap = max(4 * pred, pred + 1500)
        tl = TaskLLM(self.llm, cap)
        t = time.time()
        ans, err = None, None
        try:
            ans = mod.solve_llm(task, tl, plan)
            if ans is None and config.ALLOW_REASONING and task.reward >= 3 \
                    and mod is not solvers.coding_solver and not plan.notes.get("generic") \
                    and self.ledger.can_afford(pred * 3):
                ans = mod.solve_llm(task, tl, plan, strong=True)
        except (BudgetExhausted, TaskBudgetExceeded) as e:
            err = type(e).__name__
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
            traceback.print_exc()
        if ans is not None:
            self.save(task, ans)
            self.stats["llm_ok"] += 1
        else:
            self.stats["llm_fail"] += 1
        self.est.observe(task.task_type, pred, tl.spent)
        self.traj({"task_id": task.task_id, "type": task.task_type, "reward": task.reward,
                   "route": "llm", "pred_cost": round(pred, 1), "cost": round(tl.spent, 1),
                   "ok": ans is not None, "answer": ans, "err": err, "calls": tl.calls,
                   "secs": round(time.time() - t, 1),
                   "features": {k: v for k, v in plan.notes.items()
                                if k in ("prompt_tokens", "targets", "tests")}})
        log(f"{task.task_id}: {'OK ' + str(ans)[:30] if ans else 'FAIL ' + str(err)} "
            f"cost={tl.spent:.0f} pred={pred:.0f} spent_total={self.ledger.spent:.0f}")

    def run(self):
        log(f"tasks={len(self.tasks)} budget={self.ledger.budget} fast_path={config.FAST_PATH} "
            f"order={config.ORDER}")
        if self.tasks and not can_write(self.tasks[0].dir):
            log(f"ERROR: task directories are not writable: {self.tasks_dir}. Answers go to OUT and to this "
                "log, coding tasks cannot be repaired.")
        with ThreadPoolExecutor(max_workers=min(4, os.cpu_count() or 2)) as ex:
            analyzed = list(ex.map(self.analyze_one, self.tasks))
        log(f"phase A done in {time.time() - self.t0:.1f}s: {self.stats}")

        queue = []
        for task, mod, plan in analyzed:
            if plan.fast_answer is not None or plan.feasible is False:
                continue
            pred, _ = self.est.predict(task.task_type, plan)
            queue.append([0.0, task, mod, plan, pred])
        rank = {id(q[1]): i for i, q in enumerate(random.Random(config.ORDER_SEED).sample(queue, len(queue)))}

        def priority(q, p_ok):
            if config.ORDER == "reward":
                return q[1].reward - 1e-6 * q[4]  # при равных баллах - дешевле раньше
            if config.ORDER == "random":
                return -rank[id(q[1])]
            return q[1].reward * p_ok / q[4]

        for q in queue:
            q[0] = priority(q, self.est.predict(q[1].task_type, q[3])[1])
        queue.sort(key=lambda q: q[0], reverse=True)
        lock = threading.Lock()

        def next_item():
            with lock:
                while queue:
                    if self.ledger.exhausted or self.time_left() < 60:
                        return None
                    # пересчитываем прогноз с учётом онлайн-калибровки
                    for q in queue:
                        q[4], p_ok = self.est.predict(q[1].task_type, q[3])
                        q[0] = priority(q, p_ok)
                    queue.sort(key=lambda q: q[0], reverse=True)
                    for i, q in enumerate(queue):
                        if self.ledger.can_afford(q[4]):
                            return queue.pop(i)
                    self.stats["skipped_budget"] += len(queue)
                    queue.clear()
                return None

        def worker():
            while True:
                item = next_item()
                if item is None:
                    return
                _, task, mod, plan, pred = item
                self.solve_one(task, mod, plan, pred)

        if self.llm.base:
            threads = [threading.Thread(target=worker, daemon=True) for _ in range(config.PARALLEL)]
            for th in threads:
                th.start()
            for th in threads:
                th.join(max(1, self.time_left()))
        else:
            log("LLM_BASE_URL not set - only LLM-free solutions")
        oj_agent.shutdown()
        self.stats["llm_calls"] = self.ledger.calls
        self.stats["tokens_spent"] = round(self.ledger.spent, 1)
        self.stats["elapsed_s"] = round(time.time() - self.t0, 1)
        if self.stats["answers_lost"]:
            log(f"ERROR: {self.stats['answers_lost']} answers were not saved anywhere")
        log("SUMMARY", json.dumps(self.stats))
        return self.stats
