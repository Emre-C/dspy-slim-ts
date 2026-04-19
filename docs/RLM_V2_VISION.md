# Next-Generation RLM: The DSPy-Slim-TS Vision

The friction and tech debt surrounding `forward` (sync) vs. `aforward` (async) stem from a legacy architectural choice: **open-ended, string-based REPL execution**. When an agent relies on `eval()` or `exec()` of LLM-generated imperative code to orchestrate sub-calls, the host language struggles to fluidly map that generic string into a native asynchronous event loop.

To build the most optimal, forward-looking TypeScript implementation without legacy constraints, we must abandon the REPL paradigm natively inherited from Python. Based on recent 2026 research, here is the blueprint for **RLM v2**.

---

## 1. The Combinator Runtime ($\lambda$-RLM)
**Reference:** *The Y-Combinator for LLMs: Solving Long-Context Rot with $\lambda$-Calculus* (arXiv:2603.20105)

The original RLM gives the LLM a blank terminal ("prompt-as-environment") and asks it to write looping Python code to recursively solve sub-problems. This leads to code execution vulnerabilities, runaway loops, and the sync/async mapping nightmare. 

> [!IMPORTANT]
> **The Paradigm Shift:** We replace the open-ended REPL with a **typed functional runtime**. We stop letting the LLM write arbitrary string slicing code.

Instead of writing code, we provide the underlying structural orchestration using pre-verified deterministic combinators:
- `Split`: Partition a large string into chunks
- `Map`: Lift a recursive or neural process over a collection
- `Filter`: Retain elements satisfying a predicate
- `Reduce`: Fold a list into an aggregate

**Why this matters for `dspy-slim-ts`:** If the orchestration AST is executed by our native TypeScript engine rather than `eval()`, we eradicate the sync/async divide. `Map` naturally executes as `Promise.all()` for parallel sub-calls. `Reduce` is an `await` accumulator. Multi-turn execution becomes a seamlessly typed, purely asynchronous execution graph. The base LLM becomes a **bounded oracle restricted solely to leaf-node subproblems**.

---

## 2. Pre-Computed Deterministic Planning & Task Routing
**Reference:** *The Y-Combinator for LLMs* (arXiv:2603.20105)

Giving the LLM the tools isn't enough; relying on it to determine *how* to decompose is inefficient and prone to runaway loops. $\lambda$-RLM reveals that we can mathematically compute the optimal decomposition strategy *before* any execution begins.

> [!TIP]
> **Math-Driven Planning:** Before solving the prompt, the RLM should run a lightweight `DeterministicPlanner`. By using a cost function, it identifies the exact optimal partition size ($k^*$) and recursion maximum depth.

Furthermore, we use a single fast LLM call to classify the *Task Type* (e.g., search, pairwise, aggregation). The runtime then mounts a pre-built execution AST (e.g., `Split -> Map(M) -> Prune -> Cross` for a pairwise task) bypassing LLM-generated orchestration entirely.

---

## 3. Shared Program State & Native Handlers
**Reference:** *Enabling RLM Inference with Shared Program State* (Ellie Y. Cheng)

If we eliminate the string-based Python REPL, how does the model interact with memory?

> [!NOTE]
> By treating RLM execution as an **Effects-Handler Pattern**, the LLM yields structured intents (Effects) rather than executable code strings.

In `dspy-slim-ts`, we can expose exact typed interfaces directly into our shared TypeScript runtime environment. 
- The model yields structured requests (e.g., `Intent: ReadContext(index=0)`).
- The TypeScript orchestration loop (the Handler) resolves these effects, updates variable scopes, and hands control back.
- This entirely bypasses the brittleness of parsing nested natural code and handles memory safely and asynchronously natively.

---

## 4. Opinionated System Reinjection (Constrained Memory)
**Reference:** *Sparse Signal Loop* (Stochi)

RLMs frequently bloat or plateau when given unconstrained files (`skill_file.txt`) or persistent scratchpads, learning to optimize for the conversational judge's approval rather than actual task success. More state is not better management. 

Stochi's research proves that saving free-form diaries to the disk is an anti-pattern. The most effective memory relies on **System Reinjection**.

> [!WARNING]
> **System Reinjection over Static Files:** Instead of allowing the model to dump free-form hypothesis logs into a text file or REPL environment, `dspy-slim-ts` must enforce strict, typed memory interfaces that are reinjected as heavily constrained system messages.

We enforce a rigid schema abstraction that gets re-inserted at the start of the next turn:
```typescript
interface RLMProcessMemory {
  failure_pattern: string;
  next_check: string;
  prevented_action: string;
}
```
By forcing the lessons learned from early exploration into a tiny, opinionated loop, we prevent the RLM from carrying over ceremonial bloat and overfitting to the judge.

---

## 5. The Master Plan: No Backwards Compatibility

To achieve the best architecture and completely resolve the `forward` vs `aforward` debate:

1. **Abolish `eval()`-based orchestration:** The LLM does not write code to control recursion; it is exclusively a leaf-node oracle. Control flow is shifted back to the strict `dspy-slim-ts` runtime.
2. **Implement the `CombinatorNode` API:** We build native TS structures for `Split`, `Map`, `Filter`, `Reduce`. 
3. **Introduce the `DeterministicPlanner`:** Predict the recursion depth and cost mathematically before neural execution begins.
4. **System Reinjection Engine:** Keep state tightly constrained and typed inside the context window, killing unstructured `skill_file` generation.
5. **Enforce Async-Exclusivity:** Concurrency is structurally mapped. Because `Map` combinators map perfectly to `Promise.all`, the runtime is natively asynchronous execution of a data graph. **Therefore, a synchronous `forward` mathematically makes no sense in this paradigm.** It vanishes entirely, bypassing the legacy API and killing the tech debt.
