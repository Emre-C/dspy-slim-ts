Title: 

Content selection saved. Describe the issue below:

Description: 

[ License: CC BY 4.0](https://info.arxiv.org/help/license/index.html#licenses-available) 

arXiv:2603.20105v1 \[cs.LG\] 20 Mar 2026

# The 𝐘\\mathbf{Y}\-Combinator for LLMs:   
Solving Long-Context Rot with λ\\lambda\-Calculus

 Amartya Roy  
The School of Interdisciplinary Research   
Indian Institute of Technology (IIT) Delhi  
Robert Bosch GmbH, India  
srz248670@iitd.ac.in   
Rasul Tutunov   
Huawei Noah’s Ark Lab  
rasul.tutunov@huawei.comXiaotong Ji   
Huawei Noah’s Ark Lab   
xiaotong.ji1@h-partners.com   
Matthieu Zimmer   
Huawei Noah’s Ark Lab   
matthieu.zimmer@huawei.comHaitham Bou-Ammar   
Huawei Noah’s Ark   
UCL Centre for Artificial Intelligence  
haitham.ammar@huawei.com 

###### Abstract

LLMs are increasingly used as general-purpose reasoners, but long inputs remain bottlenecked by a fixed context window. Recursive Language Models (RLMs) address this by externalising the prompt and recursively solving subproblems. Yet existing RLMs depend on an open-ended read–eval–print loop (REPL) in which the model generates arbitrary control code, making execution difficult to verify, predict, and analyse.

We introduce λ​\-RLM\\lambda\\text{-}\\textsf{RLM}, a framework for long-context reasoning that replaces free-form recursive code generation with a typed functional runtime grounded in λ\\lambda\-calculus. It executes a compact library of pre-verified combinators and uses neural inference only on bounded leaf subproblems, turning recursive reasoning into a structured functional program with explicit control flow. We show that λ​\-RLM\\lambda\\text{-}\\textsf{RLM} admits formal guarantees absent from standard RLMs, including termination, closed-form cost bounds, controlled accuracy scaling with recursion depth, and an optimal partition rule under a simple cost model. Empirically, across four long-context reasoning tasks and nine base models, λ​\-RLM\\lambda\\text{-}\\textsf{RLM} outperforms standard RLM in 29 of 36 model-task comparisons, improves average accuracy by up to +21.9 points across model tiers, and reduces latency by up to 4.1×\\textbf{4.1}\\times. These results show that typed symbolic control yields a more reliable and efficient foundation for long-context reasoning than open-ended recursive code generation. The complete implementation of λ​\-RLMs\\lambda\\text{-}\\textsf{RLMs}, is open-sourced for the community at: [github.com/lambda-calculus-LLM/lambda-RLM](https://github.com/lambda-calculus-LLM/lambda-RLM).

## 1 Introduction

Large language models (LLMs) are increasingly used as general-purpose problem solvers (Brown et al., [2020](#bib.bib5); Yao et al., [2023a](#bib.bib37); Mower et al., [2024](#bib.bib23); Zimmer et al., [2025b](#bib.bib45); Ji et al., [2026a](#bib.bib17)), yet one of their most fundamental bottlenecks remains unchanged: _a Transformer consumes a fixed-length context window_ (Dai et al., [2019](#bib.bib10)). When inputs exceed this limit, e.g., long documents, codebases, multi-file repositories, or large collections of evidence, naïvely truncating context or relying on sliding-window prompting forces the model to “forget” early information and often breaks tasks that require global consistency or systematic evidence gathering (Liu et al., [2023](#bib.bib22); Wang et al., [2024](#bib.bib30)). In response, a growing line of work reframes long-context reasoning as inference-time scaling: rather than increasing model parameters or training new architectures, we can scale computation at inference by decomposing problems into smaller subproblems and composing their solutions (Zhou et al., [2023](#bib.bib42); Yao et al., [2023b](#bib.bib38), [a](#bib.bib37); Yang et al., [2025b](#bib.bib36)).

 

Figure 1: A summary of our results comparing λ​\-RLMs\\lambda\\text{-}\\textsf{RLMs} to base LLMs and recursive LLMs. Those results demonstrate improvements reaching +21.9 in accuracy, with 4.1×\\textbf{4.1}\\times in latency reductions.

A particularly compelling recent proposal is Recursive Language Models (RLMs), which argues that arbitrarily long user prompts should not be fed into the neural network directly (Zhang et al., [2026](#bib.bib40)). Instead, the prompt should be treated as part of an external environment that the model can interact with symbolically. Concretely, RLM initialises a programming environment (a REPL) in which the prompt is stored as a variable; the LLM then writes code to peek into the prompt, decompose it into slices, and recursively invoke itself on those slices as needed. This simple interface, prompt-as-environment plus symbolic recursion, enables models to handle inputs far beyond their native context length while retaining a standard “string-in, string-out” API.

However, RLM’s power comes with a practical cost: it relies on an LLM-driven control loop that emits and executes arbitrary code until the model decides it has finished. This open-ended REPL loop is difficult to bound and audit. In practice, it creates several failure modes that are orthogonal to the underlying reasoning task: code may not parse or may crash at runtime; recursion may be invoked excessively; intermediate outputs may be malformed; and computation may become unpredictable due to the model’s own control-flow decisions. More broadly, giving an LLM unrestricted freedom to program its own execution introduces an undesirable coupling between what the model knows and how it is allowed to search and compose evidence.

In this work, we propose λ​\-RLM\\lambda\\text{-}\\textsf{RLM}, a framework that retains the key insight of RLM, prompt-as-environment with recursive decomposition but replaces open-ended code generation with a typed, functional runtime grounded in λ\\lambda\-Calculus. λ​\-RLM\\lambda\\text{-}\\textsf{RLM} expresses all control flow through a small library of deterministic, compositional operators (e.g., Split, Map, Filter, Reduce) that are pre-verified and loaded into the REPL before execution. The base language model ℳ\\mathcal{M} is invoked only at the leaves of the recursion, on sub-prompts that are guaranteed to fit within its context window KK; all higher-level decisions i) how to split, ii) how many chunks, iii) when to stop, iv)how to compose are made by a planner and executed symbolically, without any LLM-generated code. Recursion is encoded as a fixed-point over this operator library (Section [3](#S3 "3 The 𝜆⁢"-RLM" Framework ‣ The 𝐘-Combinator for LLMs: Solving Long-Context Rot with 𝜆-Calculus")), and the planner enforces predictable execution: maximum depth d\=⌈logk∗⁡(n/τ∗)⌉d=\\lceil\\log\_{k^{\*}}(n/\\tau^{\*})\\rceil, a pre-computed number of ℳ\\mathcal{M} calls, and deterministic composition at every level. As a result, λ​\-RLM\\lambda\\text{-}\\textsf{RLM} separates _semantic reasoning_ from _structural control_: the model contributes understanding only where it is needed; at leaf sub-problems small enough to process reliably while all orchestration is handled by an auditable, deterministic controller with formal guarantees on termination, cost, and accuracy.

We choose λ\\lambda\-Calculus as our foundation because it provides a minimalist yet universal interface for hierarchical reasoning that other formalisms lack. While Finite State Machines (FSMs) are insufficient for the arbitrary recursion depths required in complex document decomposition, and Planning Domain Definition Languages (PDDL) are optimised for state-space search rather than data transformation, λ\\lambda\-Calculus treats the prompt as a first-class functional object. Crucially, by utilising fixed-point combinators (e.g., the YY\-combinator), λ\\lambda\-RLM "ties the knot" of recursion without requiring the LLM to manage function names or global state, effectively eliminating the reference errors and non-termination failures common in open-ended REPL loops.

Our design choices yield three benefits. First, λ​\-RLM\\lambda\\text{-}\\textsf{RLM} provides termination by construction under mild conditions on the splitting operator, eliminating a common class of non-termination and runaway-execution failures in agentic scaffolds. Second, it yields predictable computation: we can bound the number of oracle calls and the total work as a function of input size and the chosen decomposition policy. Third, it improves reliability by reducing the number of “critical decisions” delegated to the language model. We evaluate λ​\-RLM\\lambda\\text{-}\\textsf{RLM} on long-context task settings, using RLM as a primary baseline. In short, our contributions can be summarised as: i) We introduce λ​\-RLM\\lambda\\text{-}\\textsf{RLM}, a typed functional runtime for prompt-as-environment long-context reasoning with recursion expressed as a fixed-point over deterministic combinators; ii) We formalise an operational semantics and prove termination and cost bounds under standard size-decreasing decomposition assumptions; and iii) We empirically compare against RLM, demonstrating improved reliability and more predictable compute while improving task performance.

We validate λ​\-RLM\\lambda\\text{-}\\textsf{RLM} on four long-context task families spanning search, aggregation, pairwise reasoning, and code understanding, across nine base models and context lengths up to 128K. Compared with normal RLM, λ​\-RLM\\lambda\\text{-}\\textsf{RLM} wins in 29/36 model-task comparisons (81%\\textbf{81}\\% overall), improves average accuracy by up to +21.9 points on weak models and +18.6 points on medium models, and delivers consistent latency reductions of 3.3×\\textbf{3.3}\\times to 4.1×\\textbf{4.1}\\times. On the most structurally demanding benchmark, OOL-Pairs, the gain reaches +28.6 points with a 6.2×\\textbf{6.2}\\times speedup. These results show that constraining control flow to a typed combinator runtime not only improves predictability but also leads to substantial empirical gains over open-ended recursive code generation.

## 2 A Short Primer on λ\\lambda\-Calculus

The lambda calculus is a minimal formal language for describing computation using only _functions and functional operations_. We include a brief primer here because λ​\-RLM\\lambda\\text{-}\\textsf{RLM} uses a functional view of control flow: recursion and composition are expressed as combinations of small operators, rather than as an LLM-driven loop that generates arbitrary code.

We use Exp to denote the set of (untyped) lambda-calculus expressions, and Var to denote a countable set of variable names. The grammar is defined by:

| Exp::=x\|(λx.Exp)|(ExpExp),x∈Var.\\texttt{Exp}::=x\\ | \\ (\\lambda x.\\ \\texttt{Exp})\\ | \\ (\\texttt{Exp}\\ \\texttt{Exp}),\\ \\ \\ x\\in\\texttt{Var}. |
| ---------------------------------------------------- | ---------------------------------- | --------------------------------------------------------------- |

Intuitively, the above is saying that every expression is one of three forms: i) A variable which is a placeholder: it gains meaning when it is bound by a function or substituted during evaluation; ii) An abstraction (function definition): If e∈Expe\\in\\texttt{Exp} and x∈Varx\\in\\texttt{Var}, then λ​x.e∈Exp\\lambda x.\\ e\\in\\texttt{Exp}. This is to be read as: “ a function that takes an argument x∈Varx\\in\\texttt{Var} and returns e∈Expe\\in\\texttt{Exp}. For instance, we could define an identity function as: id\=λ​x.x\\texttt{id}=\\lambda x.\\ x, or a constant function that returns its first argument as: const\=λ​x.λ​y.x.\\texttt{const}=\\lambda x.\\ \\lambda y.\\ x.; and iii) An Application (functional call): If e1,e2∈Expe\_{1},e\_{2}\\in\\texttt{Exp}, then (e1,e2)∈Exp(e\_{1},e\_{2})\\in\\texttt{Exp}, which is to be read as “apply e1e\_{1} to e2e\_{2}”. For example, the abstraction (λx.x)y(\\lambda x.x)\\ y applies the identity function to yy. By convention, application associates to the left: e1​e2​e3≡(e1​e2)​e3e\_{1}\\ e\_{2}\\ e\_{3}\\equiv\\ (e\_{1}\\ e\_{2})\\ e\_{3}. In other words, f​a​bf\\ a\\ b means “first apply ff to aa, then apply the result to bb”. We may omit the outer parentheses when unambiguous.

Syntax tells us what expressions look like. Evaluation tells us how expressions compute. In the untyped lambda calculus, the central computational rule is β\\beta\-reduction, which formalises what it means to apply a function to an argument. If we have a function λ​x.e\\lambda x.\\ e and we apply it to an argument aa, we reduce by substituting the argument aa for the variable xx inside the body ee:

| (λx.e)a→𝛽e\[x:=a\],where e​\[x:=a\] takes e and replaces every free occurrence of x by a. (\\lambda x.\\ e)\\ a\\xrightarrow{\\beta}e\[x:=a\],\\ \\ \\text{where $e\[x:=a\]$ takes $e$ and replaces every free occurrence of $x$ by $a$. } |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |

In other words, β\\beta\-reduction is just a function application as substitution, exactly like evaluating a function call. Let us develop some examples to understand β\\beta\-reduction better:

Examples of β\\beta\-Reduction Identity. Let id\=λ​x.x\\texttt{id}=\\lambda x.\\,x. Then (λx.x)y→𝛽y(\\lambda x.\\,x)\\,y\\xrightarrow{\\beta}y: applying the identity returns its input. Constant function. Let const\=λ​x.λ​y.x\\texttt{const}=\\lambda x.\\,\\lambda y.\\,x. By left associativity, constab\=((λx.λy.x)a)b\\texttt{const}\\;a\\;b=((\\lambda x.\\,\\lambda y.\\,x)\\;a)\\;b. Reducing step by step: (λx.λy.x)a⏟→𝛽λy.a​b→𝛽a.\\underbrace{(\\lambda x.\\,\\lambda y.\\,x)\\;a}\_{\\xrightarrow{\\beta}\\;\\lambda y.\\,a}\\;b\\;\\xrightarrow{\\beta}\\;a. The outer λ\\lambda binds aa, yielding a constant function λ​y.a\\lambda y.\\,a that ignores its argument. Applying it to bb returns aa. 

#### Recursion & Fixed-Point Combinators.

λ\\lambda\-Calculus functions are anonymous, so recursion is not built-in. In Python one writes def f(...): ... f(...) ... the name f enables self-reference. Without names, the trick is _fixed points_: a value uu satisfying u\=g​(u)u=g(u) for a given function gg.

A _fixed-point combinator_ fix is a higher-order term satisfying fix​(g)\=g​(fix​(g))\\texttt{fix}(g)=g(\\texttt{fix}(g)) for all gg. Intuitively, gg is a non-recursive _recipe_ that says: “here is one step of the computation, assuming you already have a solver ff for strictly smaller sub-problems.” The combinator fix _ties the knot_, converting this one-step recipe into a genuinely recursive function by ensuring f\=g​(f)f=g(f).

In the untyped λ\\lambda\-Calculus, one concrete realisation is the Y-combinator 𝐘\\mathbf{Y}, satisfying 𝐘​g→𝛽g​(𝐘​g)\\mathbf{Y}\\,g\\xrightarrow{\\beta}g(\\mathbf{Y}\\,g)\-a fixed point of gg without any external naming mechanism.

###### Definition 1 (Fixed-Point Combinator). 

The Y-combinator enables recursion in the untyped lambda calculus: 𝐘≡λf.(λx.f(xx))(λx.f(xx))\\mathbf{Y}\\equiv\\lambda f.\\,(\\lambda x.\\,f\\,(x\\,x))\\,(\\lambda x.\\,f\\,(x\\,x)), satisfying 𝐘​g\=g​(𝐘​g)\\mathbf{Y}\\,g=g\\,(\\mathbf{Y}\\,g) for all gg.

Worked Example: Factorial via the Y-Combinator In Python, factorial calls itself by name: def fact(n): return 1 if n==0 else n\*fact(n-1). In l​a​m​b​d​a\\\\ lambda\-Calculus there are no names, so we separate the _one-step recipe_ from the recursion mechanism. Step 1: Write the recipe. Define a functional GG that takes a candidate solver ff and returns a one-step factorial procedure: G≜λ​f.λ​n.𝐢𝐟​(n\=0)​𝐭𝐡𝐞𝐧​ 1​𝐞𝐥𝐬𝐞​n⋅f​(n−1).G\\;\\triangleq\\;\\lambda f.\\,\\lambda n.\\,\\mathbf{if}\\;(n=0)\\;\\mathbf{then}\\;1\\;\\mathbf{else}\\;n\\cdot f(n-1). GG is _not_ recursive - it never calls itself. It says: “given a solver ff for smaller inputs, here is one step.” Step 2: Apply the Y-combinator. Recall 𝐘\=λg.(λx.g(xx))(λx.g(xx))\\mathbf{Y}=\\lambda g.\\,(\\lambda x.\\,g\\,(x\\,x))\\,(\\lambda x.\\,g\\,(x\\,x)). Define fact≜𝐘​G\\texttt{fact}\\triangleq\\mathbf{Y}\\,G and expand: fact \=𝐘G\=(λg.(λx.g(xx))(λx.g(xx)))G\\displaystyle=\\mathbf{Y}\\,G=\\bigl(\\lambda g.\\,(\\lambda x.\\,g\\,(x\\,x))\\,(\\lambda x.\\,g\\,(x\\,x))\\bigr)\\,G →𝛽(λx.G(xx))(λx.G(xx))\\displaystyle\\xrightarrow{\\beta}\\;(\\lambda x.\\,G\\,(x\\,x))\\,(\\lambda x.\\,G\\,(x\\,x)) (substitute g:=Gg:=G) →𝛽G​((λx.G(xx))(λx.G(xx))⏟\=𝐘​G⁣\=fact)\=G​(fact).\\displaystyle\\xrightarrow{\\beta}\\;G\\,\\bigl(\\underbrace{(\\lambda x.\\,G\\,(x\\,x))\\,(\\lambda x.\\,G\\,(x\\,x))}\_{=\\,\\mathbf{Y}\\,G\\,=\\,\\texttt{fact}}\\bigr)\\;=\\;G(\\texttt{fact}). (the knot is tied) The self-referential term (x​x)(x\\,x) is the engine: each copy of λ​x.G​(x​x)\\lambda x.\\,G(x\\,x) feeds _itself_ as the argument, producing G​(fact)G(\\texttt{fact}) \- exactly the identity 𝐘​G\=G​(𝐘​G)\\mathbf{Y}\\,G=G(\\mathbf{Y}\\,G). Step 3: Verify. Expanding G​(fact)G(\\texttt{fact}) recovers the familiar recursive definition: fact\=G​(fact)→𝛽λ​n.𝐢𝐟​(n\=0)​𝐭𝐡𝐞𝐧​ 1​𝐞𝐥𝐬𝐞​n⋅fact​(n−1).\\texttt{fact}\\;=\\;G(\\texttt{fact})\\;\\xrightarrow{\\beta}\\;\\lambda n.\\,\\mathbf{if}\\;(n=0)\\;\\mathbf{then}\\;1\\;\\mathbf{else}\\;n\\cdot\\texttt{fact}(n-1). Step 4: Trace fact​(3)\\texttt{fact}(3). Each recursive call re-triggers the same 𝐘\\mathbf{Y} machinery: fact​(3)\\displaystyle\\texttt{fact}(3) →𝛽 3⋅fact​(2)→𝛽 3⋅2⋅fact​(1)→𝛽 3⋅2⋅1⋅fact​(0)→𝛽 3⋅2⋅1⋅1\=6.\\displaystyle\\;\\xrightarrow{\\beta}\\;3\\cdot\\texttt{fact}(2)\\;\\xrightarrow{\\beta}\\;3\\cdot 2\\cdot\\texttt{fact}(1)\\;\\xrightarrow{\\beta}\\;3\\cdot 2\\cdot 1\\cdot\\texttt{fact}(0)\\;\\xrightarrow{\\beta}\\;3\\cdot 2\\cdot 1\\cdot 1=6. 

### 2.1 Core Definitions for λ​\-RLM\\lambda\\text{-}\\textsf{RLM} 

In addition to what we presented above, this section also introduces additional definitions needed for the remainder of the paper. Namely, we introduce base language models, cost functions for invoking a base model, and accuracy decays for those models as a function of the prompt’s length.

###### Definition 2 (Base Language Model). 

A base language model is a function ℳ:Σ∗→Σ∗\\mathcal{M}:\\Sigma^{\*}\\to\\Sigma^{\*} with context window K∈ℕK\\in\\mathbb{N}, such that ℳ\\mathcal{M} is only defined (or reliable) on inputs of length |P|≤K|P|\\leq K.

###### Definition 3 (Cost Function). 

The cost of invoking ℳ\\mathcal{M} on nn tokens:

| 𝒞​(n)\=cin⋅n+cout⋅n¯out\\mathcal{C}(n)=c\_{\\text{in}}\\cdot n+c\_{\\text{out}}\\cdot\\bar{n}\_{\\text{out}} | (1) |
| ------------------------------------------------------------------------------------------------------------- | --- |

where cin,coutc\_{\\text{in}},c\_{\\text{out}} are per-token prices and n¯out\\bar{n}\_{\\text{out}} is expected output length.

###### Definition 4 (Accuracy Decay). 

The accuracy of ℳ\\mathcal{M} on a prompt of length nn:

| 𝒜​(n)\=𝒜0⋅ρn/K,ρ∈(0,1\]\\mathcal{A}(n)=\\mathcal{A}\_{0}\\cdot\\rho^{\\,n/K},\\quad\\rho\\in(0,1\] | (2) |
| ---------------------------------------------------------------------------------------------------- | --- |

where 𝒜0\\mathcal{A}\_{0} is peak accuracy and ρ\\rho is the context-rot decay factor.

###### Definition 5 (Composition Operator). 

A composition operator ⊕:Σ∗×Σ∗→Σ∗\\oplus:\\Sigma^{\*}\\times\\Sigma^{\*}\\to\\Sigma^{\*} is a deterministic function that combines partial results. We define a family {⊕τ}τ∈𝒯\\{\\oplus\_{\\tau}\\}\_{\\tau\\in\\mathcal{T}} indexed by task type τ\\tau.

## 3 The λ​\-RLM\\lambda\\text{-}\\textsf{RLM} Framework

The key idea behind λ\\lambda\-RLM is simple: long-context reasoning should be recursive, but the recursion should be executed by a small trusted runtime rather than by arbitrary code written by the language model.

λ\\lambda\-RLM keeps the central insight of RLM—prompt-as-environment with recursive decomposition, but replaces open-ended code generation with a typed functional runtime. Instead of allowing the model to emit arbitrary programs, λ\\lambda\-RLM executes a fixed library of pre-verified combinators such as Split, Map, Filter, and Reduce. The base language model is used only as a bounded oracle on small leaf subproblems. In this way, λ\\lambda\-RLM separates reasoning content, which remains neural, from control flow, which becomes symbolic, deterministic, and auditable.

This design is appealing for three reasons. First, it makes execution more reliable by removing many failure modes associated with free-form code generation. Second, it makes computation predictable: once a decomposition strategy is chosen, the number of recursive calls is bounded in advance. Third, it makes the system easier to analyse formally, since recursion is expressed through a fixed functional structure rather than an open-ended loop.

### 3.1 From Open-Ended Control to a Restricted Runtime

Standard RLM operates through a REPL-style interaction. At each iteration, the model generates a code snippet from the conversation history, the REPL executes it and returns both an updated state and a standard-output string, and the output is appended to the history so the model can condition on it in the next turn:

| while True:code←ℳ(hist);(state,out)←ℰ(state,code);if state\[Final\]: break,\\textbf{while }\\texttt{True}:\\quad\\text{code}\\leftarrow\\mathcal{M}(\\text{hist});\\quad(\\text{state},\\text{out})\\leftarrow\\mathcal{E}(\\text{state},\\text{code});\\quad\\textbf{if }\\text{state}\[\\texttt{Final}\]:\\textbf{ break}, | (3) |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --- |

where the prompt lives in the environment and the model repeatedly writes code that is then executed. In more detail, the REPL provides: i) PP as a symbolic variable the LLM can reference without consuming context, ii) persistent state for intermediate results, iii) a code execution environment for programmatic decomposition, and iv) a sub-call function enabling recursive ℳ\\mathcal{M} invocations.

Indeed, this setup is powerful, but it delegates too much control to a stochastic model. The model must decide what to inspect, how to decompose the task, when to recurse, how to aggregate results, and when to stop. This can easily create an open-ended loop with no termination guarantee, no cost predictability, and a hard requirement on coding ability.

Here lies the central design choice of λ\\lambda\-RLM: we do not remove the REPL abstraction itself, but only the _open-endedness_ of what may be executed inside it. Concretely, the environment still stores the prompt externally, still exposes symbolic accessors such as peeking and slicing, and still supports recursive sub-calls to the base model. What changes is the control interface. Rather than allowing the language model to synthesise arbitrary programs token by token, we restrict execution to a small typed library of trusted combinators with known operational behaviour.

This restriction is important because it isolates the source of uncertainty. In standard RLMs, uncertainty enters twice: first through the model’s semantic judgments about the task, and second through the model’s generated control flow, which may be malformed, inefficient, or non-terminating. In λ\\lambda\-RLM, these two roles are separated. The language model is used only where neural inference is genuinely needed, namely, to solve bounded leaf subproblems. By contrast, decomposition, traversal, filtering, and aggregation are delegated to deterministic symbolic operators whose behaviour can be verified independently of the model.

Viewed differently, λ\\lambda\-RLM replaces _program synthesis as control_ with _function composition as control_. The execution trace is no longer an unbounded sequence of model-written commands, but a typed composition of operators. This shift is what makes the runtime analysable: once the decomposition rule and base threshold are fixed, the depth of recursion, the number of model calls, and the overall execution cost become explicit functions of the input size.

The resulting perspective is that long-context reasoning should be implemented as a _restricted recursive program_ with a single learned oracle, rather than as a fully model-authored agentic loop. This is the key conceptual move behind λ\\lambda\-RLM and motivates the formal definition that follows.

#### A Compact Combinator Library.

We design our combinator library to be _minimally sufficient_ for the kinds of recursive control patterns that repeatedly arise in long-context reasoning. In particular, such tasks typically require only a small set of operations: partitioning an input into manageable pieces, selectively inspecting or pruning those pieces, applying a subroutine to each retained component, and aggregating the resulting outputs into a final answer. We therefore choose a library of typed, deterministic combinators that correspond exactly to these roles. This choice is deliberate: the goal of λ\\lambda\-RLM is not to maximise expressivity at the control level, but to retain only the expressivity needed for structured decomposition while eliminating the open-ended failure modes of free-form code generation.

More concretely, the library is organised around five functional motifs. SPLIT and PEEK support decomposition and local inspection of the external prompt; MAP lifts recursive or neural processing over collections; FILTER enables symbolic selection and pruning; REDUCE, CONCAT, and CROSS provide structured aggregation and composition; and M is the only neural primitive, used exclusively on bounded leaf inputs. Together, these operators capture the dominant execution patterns underlying search, classification, aggregation, pairwise comparison, summarisation, and multi-hop composition, while keeping the runtime finite, typed, and auditable. We instantiate this principle with the compact combinator library shown in Table [1](#S3.T1 "Table 1 ‣ A Compact Combinator Library. ‣ 3.1 From Open-Ended Control to a Restricted Runtime ‣ 3 The 𝜆⁢"-RLM" Framework ‣ The 𝐘-Combinator for LLMs: Solving Long-Context Rot with 𝜆-Calculus"), where each operator is chosen by control function rather than by domain specificity. Importantly, every combinator except ℳ\\mathcal{M} is _deterministic and pre-verified_. The LLM is the only source of uncertainty.

Table 1: A compact combinator library ℒ\\mathcal{L} and examples of task-specific execution plans.

Panel A: Combinators (pre-verified, loaded into REPL)

| Combinator    | Type Signature                                                              | Description                                            |
| ------------- | --------------------------------------------------------------------------- | ------------------------------------------------------ |
| Split         | Σ∗×ℕ→\[Σ∗\]\\Sigma^{\*}\\times\\mathbb{N}\\to\[\\Sigma^{\*}\]               | Partition a string into kk contiguous chunks           |
| Peek          | Σ∗×ℕ2→Σ∗\\Sigma^{\*}\\times\\mathbb{N}^{2}\\to\\Sigma^{\*}                  | Extract a substring by start and end position          |
| Map           | (α→β)×\[α\]→\[β\](\\alpha\\to\\beta)\\times\[\\alpha\]\\to\[\\beta\]        | Apply a function to every element of a list            |
| Filter        | (α→𝔹)×\[α\]→\[α\](\\alpha\\to\\mathbb{B})\\times\[\\alpha\]\\to\[\\alpha\] | Retain elements satisfying a predicate                 |
| Reduce        | (β×β→β)×\[β\]→β(\\beta\\times\\beta\\to\\beta)\\times\[\\beta\]\\to\\beta   | Fold a list into a single value via a binary operator  |
| Concat        | \[Σ∗\]→Σ∗\[\\Sigma^{\*}\]\\to\\Sigma^{\*}                                   | Join a list of strings into one string                 |
| Cross         | \[α\]×\[β\]→\[(α,β)\]\[\\alpha\]\\times\[\\beta\]\\to\[(\\alpha,\\beta)\]   | Cartesian product of two lists                         |
| ℳ\\mathcal{M} | Σ∗→Σ∗\\Sigma^{\*}\\to\\Sigma^{\*}                                           | _Neural oracle_: invoke the base model on a sub-prompt |

Panel B: Task type, composition operator, and execution plan

| Task type  | Composition ⊕\\oplus                              | Execution plan π\\pi                                                                                                                                                                   |
| ---------- | ------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| search     | FilterBest                                        | Split→Map​(Peek)→Filter→Map​(ℳ)→Best\\textsc{Split}\\to\\textsc{Map}(\\textsc{Peek})\\to\\textsc{Filter}\\to\\textsc{Map}(\\mathcal{M})\\to\\textsc{Best}                              |
| classify   | Concat                                            | Split→Map​(ℳ)→Concat\\textsc{Split}\\to\\textsc{Map}(\\mathcal{M})\\to\\textsc{Concat}                                                                                                 |
| aggregate  | Merge                                             | Split→Map​(ℳ)→Merge\\textsc{Split}\\to\\textsc{Map}(\\mathcal{M})\\to\\textsc{Merge}                                                                                                   |
| pairwise   | Cross∘Filter\\textsc{Cross}\\circ\\textsc{Filter} | Split→Map​(ℳ)→Parse→Filter→Cross\\textsc{Split}\\to\\textsc{Map}(\\mathcal{M})\\to\\textsc{Parse}\\to\\textsc{Filter}\\to\\textsc{Cross}                                               |
| summarise  | ℳ∘Concat\\mathcal{M}\\circ\\textsc{Concat}        | Split→Map​(ℳ)→Concat→ℳ\\textsc{Split}\\to\\textsc{Map}(\\mathcal{M})\\to\\textsc{Concat}\\to\\mathcal{M}                                                                               |
| multi\_hop | ℳ∘Concat\\mathcal{M}\\circ\\textsc{Concat}        | Splitδ→Map​(Peek)→Filter→Map​(ℳ)→ℳsynth\\textsc{Split}\_{\\delta}\\to\\textsc{Map}(\\textsc{Peek})\\to\\textsc{Filter}\\to\\textsc{Map}(\\mathcal{M})\\to\\mathcal{M}\_{\\text{synth}} |

We do not claim that this library is unique or exhaustive. Indeed, it would be neither realistic nor desirable to pre-specify all combinators that may be useful for every reasoning domain. Which symbolic operators are needed can depend on the structure of the tasks under consideration. Our goal here is, therefore, more modest and more practical: we present a compact instantiation that already covers a broad range of long-context reasoning patterns, including those evaluated in the experimental section. This should be understood as an extensible basis rather than a closed vocabulary. New typed combinators can be added conservatively without altering the central λ\\lambda\-RLM principle, and we open-source the library with a lightweight interface to support such extensions.

### 3.2 Core Formulation

At the heart of λ\\lambda\-RLM is a single recursive functional program. Rather than expressing control as an open-ended REPL loop in which the language model repeatedly generates code, we express the entire controller as a fixed-point of a typed functional operator. Intuitively, this program says: if the prompt is already small enough, solve it directly with the base model; otherwise, split it into smaller pieces, solve each piece recursively, and combine the partial results using a task-specific composition rule. Formally, λ\\lambda\-RLM is defined by the lambda term:

| λ\-RLM≡fix(λf.λP.if \|P|≤τ∗ then\\displaystyle\\lambda\\text{-RLM}\\;\\equiv\\;\\texttt{fix}\\,\\Big(\\lambda f.\\,\\lambda P.\\,\\textbf{if }|P|\\leq\\tau^{\*}\\textbf{ then }               | ℳ​(P)\\displaystyle\\mathcal{M}(P) | (4) |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- | --- |
| else Reduce(⊕,Map(λpi.fpi,Split(P,k∗)))),\\displaystyle\\textbf{ else }\\textsc{Reduce}\\big(\\oplus,\\;\\textsc{Map}(\\lambda p\_{i}.\\,f\\,p\_{i},\\;\\textsc{Split}(P,k^{\*}))\\big)\\Big), |                                    |     |

where PP is the prompt stored in the external environment, k∗k^{\*} is the chosen partition size, τ∗\\tau^{\*} is the base-case threshold, and ⊕\\oplus is the task-dependent composition operator.

This term should be read from the inside out. The operator Split​(P,k∗)\\textsc{Split}(P,k^{\*}) deterministically decomposes the prompt into k∗k^{\*} sub-prompts. The higher-order combinator Map(λpi.fpi,⋅)\\textsc{Map}(\\lambda p\_{i}.\\,f\\,p\_{i},\\cdot) then applies the same recursive solver to each sub-prompt, producing a list of partial outputs. Finally, Reduce​(⊕,⋅)\\textsc{Reduce}(\\oplus,\\cdot) aggregates these outputs into a single result according to the task at hand, for example, by concatenation, merging, filtering, or synthesis.

Algorithm 1 λ​\-RLM\\lambda\\text{-}\\textsf{RLM}: Complete System

1:Prompt P∈Σ∗P\\in\\Sigma^{\*}, model ℳ\\mathcal{M} with window KK, accuracy target α∈(0,1\]\\alpha\\in(0,1\] 

2:Response Y∈Σ∗Y\\in\\Sigma^{\*} 

3:// == Phase 1: REPL Initialization == 

4:state←InitRepl​(prompt\=P)\\text{state}\\leftarrow\\textsc{InitRepl}(\\texttt{prompt}=P) ⊳\\triangleright PP stored in environment, not in context window 

5:state←RegisterLibrary​(state,ℒ)\\text{state}\\leftarrow\\textsc{RegisterLibrary}(\\text{state},\\,\\mathcal{L}) ⊳\\triangleright load pre-verified combinators into REPL 

6:state←RegisterSubCall​(state,sub\_​ℳ)\\text{state}\\leftarrow\\textsc{RegisterSubCall}(\\text{state},\\,\\texttt{sub\\\_}\\mathcal{M}) ⊳\\triangleright register ℳ\\mathcal{M} as REPL-callable leaf solver 

7:// == Phase 2: Task Detection == 

8:meta←ℰ​(state,Peek(P, 0, 500); len(P))\\text{meta}\\leftarrow\\mathcal{E}\\bigl(\\text{state},\\;\\texttt{Peek(P, 0, 500); len(P)}\\bigr) ⊳\\triangleright symbolic probe, no ℳ\\mathcal{M} call 

9:τtype←ℳ​(“Select from ​𝒯​: ”∥meta)\\tau\_{\\text{type}}\\leftarrow\\mathcal{M}\\bigl(\\text{\`\`Select from }\\mathcal{T}\\text{: ''}\\,\\|\\,\\text{meta}\\bigr) ⊳\\triangleright single ℳ\\mathcal{M} call: menu selection 

10:// == Phase 3: Dispatch == 

11:if |P|≤K|P|\\leq K then ⊳\\triangleright prompt fits in context window 

12: return sub\_​ℳ​(P)\\texttt{sub\\\_}\\mathcal{M}(P) ⊳\\triangleright direct call, no decomposition needed 

13:end if 

14:// == Phase 4: Planning (only reached if |P|\>K|P|>K) == 

15:(⊕,π)←LookupPlan​(τtype,Table​[1](#S3.T1 "Table 1 ‣ A Compact Combinator Library. ‣ 3.1 From Open-Ended Control to a Restricted Runtime ‣ 3 The 𝜆⁢"-RLM" Framework ‣ The 𝐘-Combinator for LLMs: Solving Long-Context Rot with 𝜆-Calculus"))(\\oplus,\\pi)\\leftarrow\\textsc{LookupPlan}(\\tau\_{\\text{type}},\\,\\textsc{Table}\\,\\ref{tab:combined}) 

16:(k∗,τ∗,d)←Plan​(|P|,K,α,⊕,π)(k^{\*},\\tau^{\*},d)\\leftarrow\\textsc{Plan}(|P|,K,\\alpha,\\oplus,\\pi) ⊳\\triangleright optimal split, threshold, depth 

17:// == Phase 5: Build and Execute Recursive Executor == 

18:state←ℰ​(state,Φ \= BuildExecutor(k∗, τ∗, ⊕, π, sub\_ℳ))\\text{state}\\leftarrow\\mathcal{E}\\bigl(\\text{state},\\;\\texttt{$\\Phi$ = BuildExecutor($k^{\*}$, $\\tau^{\*}$, $\\oplus$, $\\pi$, sub\\\_$\\mathcal{M}$)}\\bigr) 

19:state←ℰ​(state,result = Φ(P))\\text{state}\\leftarrow\\mathcal{E}\\bigl(\\text{state},\\;\\texttt{result = $\\Phi$(P)}\\bigr) ⊳\\triangleright single execution of Φ\\Phi in REPL 

20:return state​\[result\]\\text{state}\[\\texttt{result}\] 

The fixed-point combinator fix is what makes the definition recursive (see Section [2](#S2 "2 A Short Primer on 𝜆-Calculus ‣ The 𝐘-Combinator for LLMs: Solving Long-Context Rot with 𝜆-Calculus")). The function variable ff stands for the solver being defined itself, so the body of the term can invoke ff on each subproblem without requiring an externally named recursive procedure. In this sense, recursion is not an emergent consequence of the model deciding to call itself again; it is an explicit semantic object built into the controller.

The conditional base case |P|≤τ∗\\lvert P\\rvert\\leq\\tau^{\*} plays a crucial role. Once a sub-prompt becomes sufficiently small, the recursive decomposition stops and control is handed to the base language model ℳ\\mathcal{M}, which acts as a bounded oracle on leaf subproblems only. All higher-level control decisions - splitting, recursion, and aggregation - remain symbolic and deterministic. Crucially, the term in Equation ([4](#S3.E4 "In 3.2 Core Formulation ‣ 3 The 𝜆⁢"-RLM" Framework ‣ The 𝐘-Combinator for LLMs: Solving Long-Context Rot with 𝜆-Calculus")) is not generated by the LLM. It is constructed by a deterministic _planner_a non-neural routine that, given the input size |P||P|, context window KK, and task type, selects the parameters(k∗,τ∗,⊕)(k^{\*},\\tau^{\*},\\oplus) and instantiates the lambda term into a concrete combinator chain. This chain is then executed inside the REPL as a pre-built functional program. The planner is described in full in Algorithm [1](#alg1 "Algorithm 1 ‣ 3.2 Core Formulation ‣ 3 The 𝜆⁢"-RLM" Framework ‣ The 𝐘-Combinator for LLMs: Solving Long-Context Rot with 𝜆-Calculus") (Phase 4) and its optimality is established in Theorem [4](#Thmtheorem4 "Theorem 4 (Optimal Partition). ‣ 4 Theoretical Guarantees ‣ The 𝐘-Combinator for LLMs: Solving Long-Context Rot with 𝜆-Calculus").

Algorithm [1](#alg1 "Algorithm 1 ‣ 3.2 Core Formulation ‣ 3 The 𝜆⁢"-RLM" Framework ‣ The 𝐘-Combinator for LLMs: Solving Long-Context Rot with 𝜆-Calculus") presents the complete λ​\-RLM\\lambda\\text{-}\\textsf{RLM} system. Like the original RLM, it initialises a REPL with PP as an environment variable. Unlike the original, it replaces the open-ended while loop from Equation ([3](#S3.E3 "In 3.1 From Open-Ended Control to a Restricted Runtime ‣ 3 The 𝜆⁢"-RLM" Framework ‣ The 𝐘-Combinator for LLMs: Solving Long-Context Rot with 𝜆-Calculus")) with deterministic verifiable phases: REPL initialisation, task detection, planning, cost estimation, and a single execution of a pre-built combinator chain Φ\\Phi. Both systems share lines 1-3: the REPL is initialised, PP is stored as an environment variable, and ℳ\\mathcal{M} is registered as a sub-callable. The critical difference is Phase 5\. The original RLM enters an open-ended while loop where ℳ\\mathcal{M} generates arbitrary code each turn. λ​\-RLM\\lambda\\text{-}\\textsf{RLM} replaces this with a _single_ REPL execution of a pre-built function Φ\\Phi (Algorithm [2](#alg2 "Algorithm 2 ‣ 3.2 Core Formulation ‣ 3 The 𝜆⁢"-RLM" Framework ‣ The 𝐘-Combinator for LLMs: Solving Long-Context Rot with 𝜆-Calculus")), whose body consists entirely of combinators from ℒ\\mathcal{L}. The while loop is eliminated; recursion is handled internally by Φ\\Phi via the fixed-point combinator, with depth bounded by d\=⌈logk∗⁡(n/τ∗)⌉d=\\lceil\\log\_{k^{\*}}(n/\\tau^{\*})\\rceil.

Algorithm 2 Φ\\Phi: Recursive Executor (More Detailed in Appendix [D](#A4 "Appendix D Algorithmic Details ‣ The 𝐘-Combinator for LLMs: Solving Long-Context Rot with 𝜆-Calculus"))

1:Prompt PP from REPL state, parameters k∗,τ∗,⊕,πk^{\*},\\tau^{\*},\\oplus,\\pi 

2:Result string YY 

3:function Φ\\Phi(PP)

4: if |P|≤τ∗|P|\\leq\\tau^{\*} then 

5: q←LeafPrompt​(P,π)q\\leftarrow\\textsc{LeafPrompt}(P,\\pi) ⊳\\triangleright task-specific leaf formatting 

6: return sub\_​ℳ​(q)\\texttt{sub\\\_}\\mathcal{M}(q) ⊳\\triangleright bounded neural call on a leaf subproblem 

7: else 

8: \[P1,…,Pk∗\]←Split​(P,k∗,π)\[P\_{1},\\dots,P\_{k^{\*}}\]\\leftarrow\\textsc{Split}(P,k^{\*},\\pi) ⊳\\triangleright deterministic: exactly k∗k^{\*} chunks 

9: \[P1′,…,Pk′′\]←PruneIfNeeded​(\[P1,…,Pk∗\],π)\[P^{\\prime}\_{1},\\dots,P^{\\prime}\_{k^{\\prime}}\]\\leftarrow\\textsc{PruneIfNeeded}(\[P\_{1},\\dots,P\_{k^{\*}}\],\\pi) ⊳\\triangleright k′≤k∗k^{\\prime}\\leq k^{\*}; identity if no pruning 

10: \[R1,…,Rk′\]←Map(λpi.Φ(pi),\[P1′,…,Pk′′\])\[R\_{1},\\dots,R\_{k^{\\prime}}\]\\leftarrow\\textsc{Map}(\\lambda p\_{i}.\\,\\Phi(p\_{i}),\\;\[P^{\\prime}\_{1},\\dots,P^{\\prime}\_{k^{\\prime}}\]) ⊳\\triangleright recursive sub-calls 

11: return Reduce​(⊕,\[R1,…,Rk′\])\\textsc{Reduce}(\\oplus,\\;\[R\_{1},\\dots,R\_{k^{\\prime}}\]) ⊳\\triangleright deterministic composition 

12: end if 

13:end function 

For tasks with non-trivial structure, we provide specialised instantiations, see Appendix [D](#A4 "Appendix D Algorithmic Details ‣ The 𝐘-Combinator for LLMs: Solving Long-Context Rot with 𝜆-Calculus"). The key insight for pairwise tasks: O​(n2)O(n^{2}) pair computation is purely symbolic (zero neural cost), while only O​(n/K)O(n/K) classification calls are neural. For multi-hop search, preview-based filtering reduces the corpus before any expensive neural reading.

## 4 Theoretical Guarantees

We now establish formal properties of Algorithm [1](#alg1 "Algorithm 1 ‣ 3.2 Core Formulation ‣ 3 The 𝜆⁢"-RLM" Framework ‣ The 𝐘-Combinator for LLMs: Solving Long-Context Rot with 𝜆-Calculus"). Throughout, let n\=|P|n=|P| and let KK denote the context window of ℳ\\mathcal{M}. The recursive executor Φ\\Phi(Algorithm [2](#alg2 "Algorithm 2 ‣ 3.2 Core Formulation ‣ 3 The 𝜆⁢"-RLM" Framework ‣ The 𝐘-Combinator for LLMs: Solving Long-Context Rot with 𝜆-Calculus")) is parameterised by three quantities chosen before execution begins:

* •  
k∗≥2k^{\*}\\geq 2: the number of chunks produced by eachSplit call (the _partition size_);
* •  
τ∗≤K\\tau^{\*}\\leq K: the maximum sub-prompt length at which recursion stops and ℳ\\mathcal{M} is called directly (the_leaf threshold_);
* •  
⊕\\oplus: the composition operator used by Reduceat each level.

These parameters are selected by Phase 4 of Algorithm [1](#alg1 "Algorithm 1 ‣ 3.2 Core Formulation ‣ 3 The 𝜆⁢"-RLM" Framework ‣ The 𝐘-Combinator for LLMs: Solving Long-Context Rot with 𝜆-Calculus"). Theorems [1](#Thmtheorem1 "Theorem 1 (Termination). ‣ 4 Theoretical Guarantees ‣ The 𝐘-Combinator for LLMs: Solving Long-Context Rot with 𝜆-Calculus")\-[3](#Thmtheorem3 "Theorem 3 (Accuracy Bound). ‣ 4 Theoretical Guarantees ‣ The 𝐘-Combinator for LLMs: Solving Long-Context Rot with 𝜆-Calculus")hold for _any_ valid choice of (k∗,τ∗,⊕)(k^{\*},\\tau^{\*},\\oplus)satisfying k∗≥2k^{\*}\\geq 2 and τ∗≤K\\tau^{\*}\\leq K; Theorem [4](#Thmtheorem4 "Theorem 4 (Optimal Partition). ‣ 4 Theoretical Guarantees ‣ The 𝐘-Combinator for LLMs: Solving Long-Context Rot with 𝜆-Calculus") then derives the _cost-minimising_ k∗k^{\*} in closed form. Our results rely on the following assumptions:

###### Assumption 1 (Model Regularity). 

1. (A1)  
ℳ:Σ∗→Σ∗\\mathcal{M}:\\Sigma^{\*}\\to\\Sigma^{\*} halts on all inputs of length ≤K\\leq K in bounded time.
2. (A2)  
Every combinator in ℒ∖{ℳ}\\mathcal{L}\\setminus\\{\\mathcal{M}\\} is total and deterministic.
3. (A3)  
The cost function 𝒞:ℕ→ℝ≥0\\mathcal{C}:\\mathbb{N}\\to\\mathbb{R}\_{\\geq 0} is monotone non-decreasing: m≤n⇒𝒞​(m)≤𝒞​(n)m\\leq n\\Rightarrow\\mathcal{C}(m)\\leq\\mathcal{C}(n).
4. (A4)  
The per-call accuracy 𝒜:ℕ→(0,1\]\\mathcal{A}:\\mathbb{N}\\to(0,1\] is monotone non-increasing on \[1,K\]\[1,K\], i.e. quality degrades with input length (context rot).
5. (A5)  
The composition operator ⊕\\oplus satisfies 𝒜⊕∈(0,1\]\\mathcal{A}\_{\\oplus}\\in(0,1\], where 𝒜⊕\\mathcal{A}\_{\\oplus} is the probability that ⊕\\oplus preserves the correct answer given correct inputs.

In the first result, we prove that our algorithm, contrary to standard RLMs, indeed terminates:

###### Theorem 1 (Termination). 

Under Assumption [1](#Thmassumption1 "Assumption 1 (Model Regularity). ‣ 4 Theoretical Guarantees ‣ The 𝐘-Combinator for LLMs: Solving Long-Context Rot with 𝜆-Calculus") (A1-A2), for any input P∈Σ∗P\\in\\Sigma^{\*} with |P|\=n<∞|P|=n<\\infty, the function Φ\\Phi in Algorithm [2](#alg2 "Algorithm 2 ‣ 3.2 Core Formulation ‣ 3 The 𝜆⁢"-RLM" Framework ‣ The 𝐘-Combinator for LLMs: Solving Long-Context Rot with 𝜆-Calculus") terminates when executed in the REPL. Moreover, the total number of ℳ\\mathcal{M} invocations is exactly:

| N​(n)\=(k∗)d+1,where ​d\=⌈logk∗⁡nτ∗⌉.N(n)=(k^{\*})^{d}+1,\\quad\\text{where }d=\\left\\lceil\\log\_{k^{\*}}\\!\\frac{n}{\\tau^{\*}}\\right\\rceil. | (5) |
| -------------------------------------------------------------------------------------------------------------------------------------------------- | --- |

###### Sketch; a complete proof is in Appendix [C](#A3 "Appendix C Proofs ‣ The 𝐘-Combinator for LLMs: Solving Long-Context Rot with 𝜆-Calculus").

Define the rank r​(P′)\=⌈logk∗⁡(|P′|/τ∗)⌉r(P^{\\prime})=\\lceil\\log\_{k^{\*}}(|P^{\\prime}|/\\tau^{\*})\\rceil and proceed by strong induction. At rank 0, |P′|≤τ∗≤K|P^{\\prime}|\\leq\\tau^{\*}\\leq K, so Φ\\Phi returns sub\_​ℳ​(q)\\texttt{sub\\\_}\\mathcal{M}(q), which halts by (A1). At rank r\>0r>0, Split produces k∗k^{\*} chunks each of size ⌈|P′|/k∗⌉<|P′|\\lceil|P^{\\prime}|/k^{\*}\\rceil<|P^{\\prime}| (since k∗≥2k^{\*}\\geq 2), strictly reducing rank; each recursive call terminates by the inductive hypothesis, and Map, Reduce, PruneIfNeeded all halt by (A2). For the call count: the recursion tree has depth dd with branching factor k∗k^{\*}, giving (k∗)d(k^{\*})^{d} leaves, each invoking sub\_​ℳ\\texttt{sub\\\_}\\mathcal{M} once. Adding the single task-detection call from Algorithm [1](#alg1 "Algorithm 1 ‣ 3.2 Core Formulation ‣ 3 The 𝜆⁢"-RLM" Framework ‣ The 𝐘-Combinator for LLMs: Solving Long-Context Rot with 𝜆-Calculus") yields N​(n)\=(k∗)d+1N(n)=(k^{\*})^{d}+1. ∎

###### Remark 1. 

With termination in place, the next question is not whether λ\\lambda\-RLM halts, but how its cost scales with input size. This is important because the planner explicitly chooses k∗k^{\*} and τ∗\\tau^{\*}, and these parameters should govern a predictable computation rather than an opaque execution trace. The following theorem shows that the total cost of our algorithm satisfies a standard recursive recurrence and therefore admits a simple closed-form expression.

###### Theorem 2 (Cost Bound). 

Under Assumptions (A1-A3), the total cost of Algorithm [1](#alg1 "Algorithm 1 ‣ 3.2 Core Formulation ‣ 3 The 𝜆⁢"-RLM" Framework ‣ The 𝐘-Combinator for LLMs: Solving Long-Context Rot with 𝜆-Calculus") satisfies the recurrence:

| T​(n)\=k∗⋅T​(nk∗)+𝒞⊕​(k∗),T​(τ∗)\=𝒞​(τ∗),T(n)=k^{\*}\\cdot T\\!\\left(\\frac{n}{k^{\*}}\\right)+\\mathcal{C}\_{\\oplus}(k^{\*}),\\qquad T(\\tau^{\*})=\\mathcal{C}(\\tau^{\*}), | (6) |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --- |

with a closed-form solution:

| T​(n)≤n​k∗τ∗​𝒞​(τ∗)+𝒞⊕​(k∗)⋅\[n​k∗−τ∗τ∗​(k∗−1)\],T(n)\\leq\\frac{nk^{\*}}{\\tau^{\*}}\\mathcal{C}(\\tau^{\*})+\\mathcal{C}\_{\\oplus}(k^{\*})\\cdot\\left\[\\frac{nk^{\*}-\\tau^{\*}}{\\tau^{\*}(k^{\*}-1)}\\right\], | (7) |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --- |

where d\=⌈logk∗⁡(n/τ∗)⌉d=\\lceil\\log\_{k^{\*}}(n/\\tau^{\*})\\rceil. When ⊕\\oplus is purely symbolic, 𝒞⊕\=0\\mathcal{C}\_{\\oplus}=0 and T​(n)≤n​k∗τ∗⋅𝒞​(τ∗)T(n)\\leq\\frac{nk^{\*}}{\\tau^{\*}}\\cdot\\mathcal{C}(\\tau^{\*}).

###### Sketch; a complete proof is in Appendix [C](#A3 "Appendix C Proofs ‣ The 𝐘-Combinator for LLMs: Solving Long-Context Rot with 𝜆-Calculus").

Unroll the recurrence dd times: T​(n)\=(k∗)d⋅T​(n/(k∗)d)+\[1+k∗+…+(k∗)d−1\]⋅𝒞⊕​(k∗)T(n)=(k^{\*})^{d}\\cdot T(n/(k^{\*})^{d})+\\left\[1+k^{\*}+\\ldots+(k^{\*})^{d-1}\\right\]\\cdot\\mathcal{C}\_{\\oplus}(k^{\*}). At depth d\=⌈logk∗⁡(n/τ∗)⌉d=\\lceil\\log\_{k^{\*}}(n/\\tau^{\*})\\rceil, the sub-problem size reaches τ∗\\tau^{\*}, hitting the base case T​(τ∗)\=𝒞​(τ∗)T(\\tau^{\*})=\\mathcal{C}(\\tau^{\*}). Substituting and noting (k∗)d∈\[n/τ∗,n​k∗/τ∗\](k^{\*})^{d}\\in\\left\[n/\\tau^{\*},nk^{\*}/\\tau^{\*}\\right\] gives T​(n)≤n​k∗τ∗⋅𝒞​(τ∗)+\[n​k∗−τ∗τ∗​(k∗−1)\]⋅𝒞⊕​(k∗)T(n)\\leq\\frac{nk^{\*}}{\\tau^{\*}}\\cdot\\mathcal{C}(\\tau^{\*})+\\left\[\\frac{nk^{\*}-\\tau^{\*}}{\\tau^{\*}(k^{\*}-1)}\\right\]\\cdot\\mathcal{C}\_{\\oplus}(k^{\*}). The first term counts all leaf β\\beta\-reductions; the second counts one composition step per level. Both are deterministic functions of nn, k∗k^{\*}, τ∗\\tau^{\*}, and the pricing constants - hence T​(n)T(n) is computable _before_ any REPL execution (line 18 of Algorithm [1](#alg1 "Algorithm 1 ‣ 3.2 Core Formulation ‣ 3 The 𝜆⁢"-RLM" Framework ‣ The 𝐘-Combinator for LLMs: Solving Long-Context Rot with 𝜆-Calculus")). ∎

###### Remark 2. 

While Theorem [2](#Thmtheorem2 "Theorem 2 (Cost Bound). ‣ 4 Theoretical Guarantees ‣ The 𝐘-Combinator for LLMs: Solving Long-Context Rot with 𝜆-Calculus") shows that recursive execution is computationally predictable, efficiency alone is not enough: the central question is whether decomposition preserves correctness. The next theorem addresses this directly. It shows that, under assumptions on bounded-input leaf accuracy and compositional reliability, the end-to-end accuracy of λ\\lambda\-RLM decays in a controlled way with depth, and can therefore compare favourably to a direct ℳ\\mathcal{M} call on inputs whose length far exceeds the native context window.

###### Theorem 3 (Accuracy Bound). 

Under Assumptions (A4-A5), letd\=⌈logk∗⁡(n/τ∗)⌉d=\\lceil\\log\_{k^{\*}}(n/\\tau^{\*})\\rceil. Sincenτ∗≤(k∗)d≤n​k∗τ∗\\frac{n}{\\tau^{\*}}\\leq(k^{\*})^{d}\\leq\\frac{n\\,k^{\*}}{\\tau^{\*}}, the end-to-end accuracy of λ​\-RLM\\lambda\\text{-}\\textsf{RLM} satisfies:

1. (i)  
If d\=0d=0 (input fits in window):𝒜λ​\-RLM​(n)\=𝒜​(τ∗)\\;\\mathcal{A}\_{\\lambda\\text{-}\\textsf{RLM}}(n)=\\mathcal{A}(\\tau^{\*}).
2. (ii)  
If d≥1d\\geq 1 and𝒜​(τ∗)<1\\mathcal{A}(\\tau^{\*})<1 (the non-trivial case):  
| 𝒜λ​\-RLM​(n)≥𝒜​(τ∗)n​k∗/τ∗⋅𝒜⊕d.\\mathcal{A}\_{\\lambda\\text{-}\\textsf{RLM}}(n)\\;\\geq\\;\\mathcal{A}(\\tau^{\*})^{\\,nk^{\*}/\\tau^{\*}}\\cdot\\mathcal{A}\_{\\oplus}^{\\,d}. | (8) |  
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --- |
3. (iii)  
If 𝒜​(τ∗)\=1\\mathcal{A}(\\tau^{\*})=1(perfect leaf accuracy):𝒜λ​\-RLM​(n)≥𝒜⊕d\\;\\mathcal{A}\_{\\lambda\\text{-}\\textsf{RLM}}(n)\\geq\\mathcal{A}\_{\\oplus}^{\\,d}.
4. (iv)  
For decomposable tasks (𝒜⊕\=1\\mathcal{A}\_{\\oplus}=1, independent sub-queries): per-query accuracy is 𝒜​(τ∗)\\mathcal{A}(\\tau^{\*}), constant in nn.

In contrast, direct inference achieves𝒜direct​(n)\=𝒜0⋅ρn/K\\mathcal{A}\_{\\textup{direct}}(n)=\\mathcal{A}\_{0}\\cdot\\rho^{\\,n/K} for ρ∈(0,1)\\rho\\in(0,1).

###### Sketch; a complete proof is in Appendix [C](#A3 "Appendix C Proofs ‣ The 𝐘-Combinator for LLMs: Solving Long-Context Rot with 𝜆-Calculus").

By induction on depth dd. At d\=0d=0, |P|≤τ∗|P|\\leq\\tau^{\*} andΦ\\Phi reduces to a single ℳ\\mathcal{M} call with accuracy𝒜​(τ∗)\\mathcal{A}(\\tau^{\*}); no composition is involved. At depthd≥1d\\geq 1, correctness in the worst case requires (i) all (k∗)d(k^{\*})^{d} leaf calls to return correct results, each with probability 𝒜​(τ∗)\\mathcal{A}(\\tau^{\*}), and (ii) all dd composition levels to preserve correctness, each with probability 𝒜⊕\\mathcal{A}\_{\\oplus}. By conditional independence of leaf evaluations on disjoint chunks, the joint probability is 𝒜​(τ∗)(k∗)d⋅𝒜⊕d\\mathcal{A}(\\tau^{\*})^{(k^{\*})^{d}}\\cdot\\mathcal{A}\_{\\oplus}^{d}. For decomposable tasks where each sub-query is answered by a single leaf, and ⊕\\oplus is deterministic (𝒜⊕\=1\\mathcal{A}\_{\\oplus}=1), the per-query accuracy is simply 𝒜​(τ∗)\\mathcal{A}(\\tau^{\*})\-constant in nn. Re-expressing the worst-case bound for d≥1d\\geq 1 as(n/τ∗)logk∗⁡𝒜​(τ∗)⋅𝒜⊕d(n/\\tau^{\*})^{\\log\_{k^{\*}}\\!\\mathcal{A}(\\tau^{\*})}\\cdot\\mathcal{A}\_{\\oplus}^{d} reveals Θ​(n−c)\\Theta(n^{-c}) power-law decay, strictly slower than Θ​(ρn/K)\\Theta(\\rho^{n/K}). ∎

###### Remark 3. 

With the recursive cost now characterised, we can move from analysis to design. In particular, the split factor kk should not be viewed as a free engineering knob: it directly controls the trade-off between leaf-level processing cost and per-level composition overhead. The following theorem makes this precise and yields an explicit cost-minimising choice of partition size.

###### Theorem 4 (Optimal Partition). 

Under Assumption (A3), for cost function 𝒞​(n)\=cin⋅n+cout⋅n¯out\\mathcal{C}(n)=c\_{\\textup{in}}\\cdot n+c\_{\\textup{out}}\\cdot\\bar{n}\_{\\textup{out}} and constant per-level composition cost 𝒞⊕​(k)\=c⊕⋅k\\mathcal{C}\_{\\oplus}(k)=c\_{\\oplus}\\cdot k, the cost-minimizing partition size for recurrence ([6](#S4.E6 "In Theorem 2 (Cost Bound). ‣ 4 Theoretical Guarantees ‣ The 𝐘-Combinator for LLMs: Solving Long-Context Rot with 𝜆-Calculus")) is k∗\=2k^{\*}=2

###### Sketch; a complete proof is in Appendix [C](#A3 "Appendix C Proofs ‣ The 𝐘-Combinator for LLMs: Solving Long-Context Rot with 𝜆-Calculus").

From ([7](#S4.E7 "In Theorem 2 (Cost Bound). ‣ 4 Theoretical Guarantees ‣ The 𝐘-Combinator for LLMs: Solving Long-Context Rot with 𝜆-Calculus")), the total cost decomposes into a leaf term (n/τ)⋅𝒞​(τ)(n/\\tau)\\cdot\\mathcal{C}(\\tau) and a composition term logk⁡(n/τ)⋅c⊕⋅k\\log\_{k}(n/\\tau)\\cdot c\_{\\oplus}\\cdot k. The upper-bound on the expression T(n)̨T(n\\k{)} can be represented as the following expression k2​(α+β)−k​(α+γ)k−1\\frac{k^{2}(\\alpha+\\beta)-k(\\alpha+\\gamma)}{k-1}, where α\=n⋅𝒞​(τ∗)τ∗​β\=c⊕⋅nτ∗\\alpha=\\frac{n\\cdot\\mathcal{C}(\\tau^{\*})}{\\tau^{\*}}\\ \\beta=\\frac{c\_{\\oplus}\\cdot n}{\\tau^{\*}} and γ\=c⊕\\gamma=c\_{\\oplus}. The minimiser of this expression with respect to kk is the smallest integer larger than ⌈1+1−α+γα+β⌉\\lceil 1+\\sqrt{1-\\frac{\\alpha+\\gamma}{\\alpha+\\beta}}\\rceil. It is easy to see that k∗\=2k^{\*}=2 is such integer.

∎

###### Remark 4. 

We next specialise the accuracy bound to its asymptotic implications as the input length grows. This makes the contrast with direct inference especially clear. In particular, the recursive structure of λ\\lambda\-RLM converts the exponential dependence on n/Kn/K into a depth-dependent composition effect, yielding polynomial decay in general and constant accuracy in the ideal decomposable setting.

###### Corollary 5 (Scaling Laws). 

Fix α\\alpha and ℳ\\mathcal{M} with window KK. As n→∞n\\to\\infty:

1. (i)  
Direct ℳ\\mathcal{M}: 𝒜direct​(n)\=Θ​(ρn/K)→0\\mathcal{A}\_{\\textup{direct}}(n)=\\Theta(\\rho^{n/K})\\to 0 exponentially.
2. (ii)  
λ​\-RLM\\lambda\\text{-}\\textsf{RLM} (worst case): 𝒜λ​\-RLM​(n)\=Ω​(n−c)\\mathcal{A}\_{\\lambda\\text{-}\\textsf{RLM}}(n)=\\Omega(n^{-c}) for c\=−logk∗⁡(𝒜​(τ∗)⋅𝒜⊕)\>0c=-\\log\_{k^{\*}}(\\mathcal{A}(\\tau^{\*})\\cdot\\mathcal{A}\_{\\oplus})>0, i.e. power-law decay.
3. (iii)  
λ​\-RLM\\lambda\\text{-}\\textsf{RLM} (decomposable tasks, 𝒜⊕\=1\\mathcal{A}\_{\\oplus}=1): 𝒜λ​\-RLM​(n)≥𝒜​(τ∗)\\mathcal{A}\_{\\lambda\\text{-}\\textsf{RLM}}(n)\\geq\\mathcal{A}(\\tau^{\*}), constant in nn.

## 5 Experiments

The primary objective of our experimental evaluation is to determine whether a restricted, typed functional runtime provides a more reliable and efficient foundation for long-context reasoning than existing neural or stochastic methods. We compareλ​\-RLM\\lambda\\text{-}\\textsf{RLM} against two baseline paradigms:

#### Direct LLM inference.

The entire prompt is fed to ℳ\\mathcal{M} in a single call. When the input length nn exceeds the model’s context window KK, one of two fallbacks is used depending on the model: either the input is truncated to the first KK tokens (with a corresponding expected drop in recall), or the run is marked as a failure with accuracy 0%0\\%. We report which fallback applies for each model in Table [2](#S5.T2 "Table 2 ‣ Normal RLM. ‣ 5 Experiments ‣ The 𝐘-Combinator for LLMs: Solving Long-Context Rot with 𝜆-Calculus"). This baseline represents the ceiling of what a single Transformer pass can achieve and exposes the cost of context rot as nn grows.

#### Normal RLM.

As per (Zhang et al., [2026](#bib.bib40)), the model writes arbitrary Python in an open-ended REPL loop to decompose and recurse over the prompt. Unlike P1, this approach can process inputs of any length, since the prompt lives in the REPL environment and only metadata or sub-prompts enter the context window.Both P2 and our method (P3: λ​\-RLM\\lambda\\text{-}\\textsf{RLM}) handle inputs far exceeding KK by design - the prompt is stored as a REPL variable and accessed symbolically. The key distinction is _how_ the REPL is used: P2 lets the LLM generate arbitrary code at each turn; P3 executes a pre-built combinator chain constructed by the planner. Our evaluation is structured to test two specific hypotheses:

* •  
The Scale-Substitution Hypothesis: We hypothesise that formal control structures can effectively substitute for raw model parameter scale, enabling "weak" tier models (e.g., 8B) to match or exceed the performance of "strong" tier models (e.g., 70B+) that lack structured orchestration; and
* •  
The Efficiency and Predictability Hypothesis: We posit that replacing multi-turn, stochastic REPL loops with a single, deterministic combinator chain will yield significant reductions in wall-clock latency and execution variance.

Models. We select three families with weak/medium/strong tiers to test the interaction between model capability and scaffold design (Table [2](#S5.T2 "Table 2 ‣ Normal RLM. ‣ 5 Experiments ‣ The 𝐘-Combinator for LLMs: Solving Long-Context Rot with 𝜆-Calculus")). All models are open-weight and served via vLLM. For both Normal RLM and λ\\lambda\-RLM, each configuration uses the same model as both root and leaf.

Table 2: Model families and strength tiers.

| Family  | Property | Weak            | Medium        | Strong          |
| ------- | -------- | --------------- | ------------- | --------------- |
| Qwen3   | Model    | Qwen3-8B        | Qwen3-32B     | Qwen3-235B-A22B |
| Context | 32K      | 128K            | 128K          |                 |
| Llama   | Model    | Llama-3.1-8B    | Llama-3.3-70B | Llama-3.1-405B  |
| Context | 128K     | 128K            | 128K          |                 |
| Mistral | Model    | Mistral-7B-v0.3 | Mixtral-8x22B | Codestral-22B   |
| Context | 32K      | 64K             | 32K           |                 |

Tasks & Scaling Protocols.To rigorously evaluate the versatility of λ​\-RLMs\\lambda\\text{-}\\textsf{RLMs}, we utilise a benchmark suite derived from (Zhang et al., [2026](#bib.bib40)) that spans a spectrum of computational complexities from O​(1)O(1) to O​(n2)O(n^{2}). This diversity ensures that our framework is tested against the varied structural demands found in long-context reasoning—from simple needle-retrieval to complex quadratic cross-referencing; see Table [3](#S5.T3 "Table 3 ‣ Normal RLM. ‣ 5 Experiments ‣ The 𝐘-Combinator for LLMs: Solving Long-Context Rot with 𝜆-Calculus") for more details. Furthermore, to validate our robust scaling hypothesis, each benchmark is executed across varying context-length buckets: {8​K,16​K,32​K,64​K,128​K}\\{8\\text{K},16\\text{K},32\\text{K},64\\text{K},128\\text{K}\\}. This graduated approach allows us to measure the onset of "context rot", i.e., the exponential decay in accuracy typically observed as standard Transformers approach their native context limits. We report macro-averages across all non-empty buckets to provide a stable, holistic metric of end-to-end reliability. This protocol explicitly highlights how the power-law decay of λ​\-RLM\\lambda\\text{-}\\textsf{RLM} compares to the exponential failure of direct inference as input size grows.

Table 3: Benchmark tasks by complexity.

| Task                                                                                             | Complexity     | Tokens   | Metric | Instances | λ​\-RLM\\lambda\\text{-}\\textsf{RLM} plan π\\pi                                                                                                     |
| ------------------------------------------------------------------------------------------------ | -------------- | -------- | ------ | --------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| S-NIAH                                                                                           | O​(1)O(1)      | 8K-128K  | F1     | 100       | Split→Map​(Peek)→Filter→Map​(ℳ)\\textsc{Split}\\!\\to\\!\\textsc{Map}(\\textsc{Peek})\\!\\to\\!\\textsc{Filter}\\!\\to\\!\\textsc{Map}(\\mathcal{M}) |
| OOLONG                                                                                           | O​(n)O(n)      | 8K-128K  | Score  | 50        | Split→Map​(ℳ)→Merge\\textsc{Split}\\!\\to\\!\\textsc{Map}(\\mathcal{M})\\!\\to\\!\\textsc{Merge}                                                     |
| OOL-Pairs 111( Refer D.1\. OOLONG-Pairs Benchmark from paper (Zhang et al., [2026](#bib.bib40))) | O​(n2)O(n^{2}) | 8K-128K  | F1     | 20        | Split→Map​(ℳ)→Parse→Cross\\textsc{Split}\\!\\to\\!\\textsc{Map}(\\mathcal{M})\\!\\to\\!\\textsc{Parse}\\!\\to\\!\\textsc{Cross}                      |
| CodeQA                                                                                           | Variable       | 23K-4.2M | Acc    | 23        | Splitδ→Map​(ℳ)→Best\\textsc{Split}\_{\\delta}\\!\\to\\!\\textsc{Map}(\\mathcal{M})\\!\\to\\!\\textsc{Best}                                           |

Prompting Strategies. For (P1) the prompt is fed to the model in a single call. In the case of (P2), we use the original system of (Zhang et al., [2026](#bib.bib40)) where the LLM generates arbitrary Python in a REPL loop. Here, we use the prompts from Appendix C of (Zhang et al., [2026](#bib.bib40)). Finally, for λ​\-RLM\\lambda\\text{-}\\textsf{RLM}, the planner constructs a combinator chain and executes it once in the REPL. For P3, the planner uses the accuracy target α\=0.80\\alpha=0.80 and per-model pricing constants from the respective API providers. All open-weight models are called through open source API calls. Each configuration is run twice, and the results are averaged. We measure task-level accuracy/F1, wall-clock latency, and the number of LLM calls per instance.

### 5.1 Main Results

Table [4](#S5.T4 "Table 4 ‣ 5.1 Main Results ‣ 5 Experiments ‣ The 𝐘-Combinator for LLMs: Solving Long-Context Rot with 𝜆-Calculus") presents accuracy across all 108 configurations (3 paradigms ×\\times 9 models ×\\times 4 tasks).

Table 4: Accuracy (%) across all paradigms, models, and tasks. Macro-averaged across context-length buckets. Best per column: λ​\-RLM\\lambda\\text{-}\\textsf{RLM} wins / Normal RLM wins.

Qwen3 Llama Mistral Task Paradigm 8B 32B 235B 8B 70B 405B 7B 8x22B Cdsrl S-NIAH P1: Direct 3.2 18.4 31.7 4.1 22.6 35.2 2.8 19.1 14.3 P2: RLM 8.4 28.3 46.8 10.2 31.5 52.4 6.1 26.4 29.7 P3: λ​\-RLM\\lambda\\text{-}\\textsf{RLM} 24.6 41.8 51.3 22.1 44.2 49.8 18.5 38.6 40.2 OOLONG P1: Direct 12.5 31.2 44.0 14.3 34.8 47.1 10.8 28.5 22.6 P2: RLM 24.1 42.7 56.5 22.6 45.3 63.8 18.3 38.9 48.2 P3: λ​\-RLM\\lambda\\text{-}\\textsf{RLM} 48.3 62.5 68.4 45.7 61.2 61.7 40.2 55.8 45.6 OOL- Pairs P1: Direct 0.1 0.3 0.1 0.1 0.4 0.2 0.0 0.2 0.1 P2: RLM 4.2 18.6 38.4 3.8 21.3 42.7 2.1 14.5 22.8 P3: λ​\-RLM\\lambda\\text{-}\\textsf{RLM} 34.8 48.2 61.5 31.6 50.7 64.3 28.4 44.1 47.6 CodeQA P1: Direct 8.7 20.4 24.0 9.2 22.1 26.3 7.4 18.6 21.2 P2: RLM 18.5 36.8 58.6 16.7 46.4 62.1 12.3 32.4 49.3 P3: λ​\-RLM\\lambda\\text{-}\\textsf{RLM} 35.2 47.1 54.2 32.8 43.8 55.7 27.6 42.5 44.8 AVG P1: Direct 6.1 17.6 25.0 6.9 20.0 27.2 5.3 16.6 14.6 P2: RLM 13.8 31.6 50.1 13.3 36.1 55.3 9.7 28.1 37.5 P3: λ​\-RLM\\lambda\\text{-}\\textsf{RLM} 35.7 49.9 58.9 33.1 49.9 57.9 28.7 45.3 44.6 

Table 5: Latency (seconds) across all configurations. Macro-averaged across context-length buckets. Direct LLM is fastest (single call, no scaffold) but has the lowest accuracy (Table [4](#S5.T4 "Table 4 ‣ 5.1 Main Results ‣ 5 Experiments ‣ The 𝐘-Combinator for LLMs: Solving Long-Context Rot with 𝜆-Calculus")). Among the two recursive paradigms, λ​\-RLM\\lambda\\text{-}\\textsf{RLM} is faster than Normal RLM in every cell.

Qwen3 Llama Mistral Task Paradigm 8B 32B 235B 8B 70B 405B 7B 8x22B Cdsrl S-NIAH P1: Direct 12.3 18.7 42.1 11.8 21.4 48.6 10.5 20.2 16.8 P2: RLM 164.3 142.8 98.6 178.2 128.4 86.3 195.7 155.1 120.4 P3: λ​\-RLM\\lambda\\text{-}\\textsf{RLM} 45.9 38.2 31.4 48.7 35.6 28.1 52.3 41.8 36.5 OOLONG P1: Direct 8.4 14.2 35.8 7.9 16.1 40.3 7.1 15.4 12.6 P2: RLM 241.6 198.3 125.7 258.4 182.5 108.2 284.1 215.3 168.7 P3: λ​\-RLM\\lambda\\text{-}\\textsf{RLM} 62.4 51.7 42.3 66.8 48.2 38.5 71.5 56.4 48.1 OOL- Pairs P1: Direct 6.2 10.8 28.4 5.8 12.3 32.1 5.1 11.7 9.4 P2: RLM 312.5 264.7 178.3 338.1 241.8 156.4 365.2 288.6 224.5 P3: λ​\-RLM\\lambda\\text{-}\\textsf{RLM} 48.6 41.2 34.7 52.1 38.4 30.8 55.8 44.6 38.2 CodeQA P1: Direct 15.6 24.3 52.7 14.8 27.6 58.4 13.2 25.8 20.1 P2: RLM 198.4 168.2 112.5 215.6 154.3 98.7 232.8 182.4 145.6 P3: λ​\-RLM\\lambda\\text{-}\\textsf{RLM} 72.3 58.6 46.8 76.4 54.2 42.1 81.5 63.7 54.3 AVG P1: Direct 10.6 17.0 39.8 10.1 19.4 44.9 9.0 18.3 14.7 P2: RLM 229.2 193.5 128.8 247.6 176.8 112.4 269.5 210.4 164.8 P3: λ​\-RLM\\lambda\\text{-}\\textsf{RLM} 57.3 47.4 38.8 61.0 44.1 34.9 65.3 51.6 44.3 

#### λ​\-RLM\\lambda\\text{-}\\textsf{RLM} wins 29 of 36 accuracy cells.

Across all model-task combinations, λ​\-RLM\\lambda\\text{-}\\textsf{RLM} achieves the highest accuracy in 81% of cases (Table [6](#S5.T6 "Table 6 ‣ 𝜆⁢"-RLM" wins 29 of 36 accuracy cells. ‣ 5.1 Main Results ‣ 5 Experiments ‣ The 𝐘-Combinator for LLMs: Solving Long-Context Rot with 𝜆-Calculus")). The wins are concentrated at the weak and medium tiers, where the coding bottleneck is most severe. At the strong tier, the win rate drops to 50%, indicating that powerful code-generating models can partially compensate for the lack of formal structure.

Table 6: Win/Loss count by model tier (accuracy).

| Model Tier         | λ​\-RLM\\lambda\\text{-}\\textsf{RLM} wins | RLM wins | Win rate |
| ------------------ | ------------------------------------------ | -------- | -------- |
| Weak (8B / 7B)     | 12 / 12                                    | 0 / 12   | 100%     |
| Medium (32B-8x22B) | 11 / 12                                    | 1 / 12   | 92%      |
| Strong (235B+)     | 6 / 12                                     | 6 / 12   | 50%      |
| All tiers          | 29 / 36                                    | 7 / 36   | 81%      |

The advantage grows with task complexity. Table [7](#S5.T7 "Table 7 ‣ 𝜆⁢"-RLM" wins 29 of 36 accuracy cells. ‣ 5.1 Main Results ‣ 5 Experiments ‣ The 𝐘-Combinator for LLMs: Solving Long-Context Rot with 𝜆-Calculus") breaks down the accuracy gain by task. The largest improvement occurs on OOLONG-Pairs (+28.6+28.6 pp), the O​(n2)O(n^{2}) task where the quadratic cross-product is handled symbolically in λ​\-RLM\\lambda\\text{-}\\textsf{RLM} but must be computed neurally in Normal RLM. Conversely, the smallest gain is on CodeQA (+10.8+10.8 pp), where ad-hoc code generation by strong models enables creative strategies (multi-pass reading, function-level chunking) that the fixed combinator library cannot express.

Table 7: λ​\-RLM\\lambda\\text{-}\\textsf{RLM} improvement by task complexity. Δ\\DeltaAcc = avg across 9 models.

| Task      | Complexity     | P2: RLM | P3: λ​\-RLM\\lambda\\text{-}\\textsf{RLM} | Δ\\DeltaAcc | Speedup     | RLM wins |
| --------- | -------------- | ------- | ----------------------------------------- | ----------- | ----------- | -------- |
| S-NIAH    | O​(1)O(1)      | 17.0    | 36.7                                      | +19.7       | 3.6×\\times | 1 / 9    |
| OOLONG    | O​(n)O(n)      | 36.7    | 55.0                                      | +18.3       | 4.2×\\times | 2 / 9    |
| OOL-Pairs | O​(n2)O(n^{2}) | 17.1    | 45.7                                      | +28.6       | 6.2×\\times | 0 / 9    |
| CodeQA    | Variable       | 33.7    | 44.5                                      | +10.8       | 3.1×\\times | 4 / 9    |
| All       | 26.1           | 45.5    | +19.4                                     | 4.0×\\times | 7 / 36      |          |

λ​\-RLM\\lambda\\text{-}\\textsf{RLM} is 𝟑\\mathbf{3}\-𝟔×\\mathbf{6\\times} faster than Normal RLM. Latency improvements are consistent across all models and tasks (Table [5](#S5.T5 "Table 5 ‣ 5.1 Main Results ‣ 5 Experiments ‣ The 𝐘-Combinator for LLMs: Solving Long-Context Rot with 𝜆-Calculus")), with the largest speedup on OOLONG-Pairs (6.2×6.2\\times). This is a direct consequence of eliminating the open-ended REPL loop: λ​\-RLM\\lambda\\text{-}\\textsf{RLM} executes a single pre-built combinator chain, while Normal RLM may iterate 5-12 turns of LLM-generated code. The latency advantage also exhibits lower variance-Normal RLM’s max/min latency ratio across instances is 8.9×8.9\\times, while λ​\-RLM\\lambda\\text{-}\\textsf{RLM}’s is 4.3×4.3\\times.

Where Normal RLM wins. Table [8](#S5.T8 "Table 8 ‣ 𝜆⁢"-RLM" wins 29 of 36 accuracy cells. ‣ 5.1 Main Results ‣ 5 Experiments ‣ The 𝐘-Combinator for LLMs: Solving Long-Context Rot with 𝜆-Calculus") lists the 7 cells where Normal RLM outperforms λ​\-RLM\\lambda\\text{-}\\textsf{RLM}. All involve either strong coding models (Llama-405B, Codestral-22B) or the CodeQA task, which benefits from free-form repository navigation. In these cases, the LLM’s ability to write creative, task-specific code; multi-pass reading with backtracking, code-aware chunking by functions, adaptive batch sizing; outweighs the reliability and speed benefits of fixed combinators.

Table 8: The 7 cells where Normal RLM outperforms λ​\-RLM\\lambda\\text{-}\\textsf{RLM}.

| Task   | Model         | Tier   | RLM  | λ​\-RLM\\lambda\\text{-}\\textsf{RLM} | Why RLM wins              |
| ------ | ------------- | ------ | ---- | ------------------------------------- | ------------------------- |
| S-NIAH | Llama-405B    | Strong | 52.4 | 49.8                                  | Creative regex search     |
| OOLONG | Llama-405B    | Strong | 63.8 | 61.7                                  | Adaptive batch sizing     |
| OOLONG | Codestral-22B | Strong | 48.2 | 45.6                                  | Optimal iteration code    |
| CodeQA | Qwen3-235B    | Strong | 58.6 | 54.2                                  | Free-form repo navigation |
| CodeQA | Llama-70B     | Medium | 46.4 | 43.8                                  | File-level decomposition  |
| CodeQA | Llama-405B    | Strong | 62.1 | 55.7                                  | Multi-pass backtracking   |
| CodeQA | Codestral-22B | Strong | 49.3 | 44.8                                  | Code-aware chunking       |

Moreover, Table [9](#S5.T9 "Table 9 ‣ 𝜆⁢"-RLM" wins 29 of 36 accuracy cells. ‣ 5.1 Main Results ‣ 5 Experiments ‣ The 𝐘-Combinator for LLMs: Solving Long-Context Rot with 𝜆-Calculus") summarises the five comparisons that directly test our hypotheses.

Table 9: Targeted comparisons. All values are averages across the 4 tasks.

| #  | Comparison                                                                                  | Acc (%)      | Lat (s)   | Verdict                                                         |
| -- | ------------------------------------------------------------------------------------------- | ------------ | --------- | --------------------------------------------------------------- |
| C1 | λ​\-RLM\\lambda\\text{-}\\textsf{RLM} (8B) vs RLM (8B)                                      | 35.7 vs 13.8 | 57 vs 229 | +21.9 pp, 4.0×\\times faster                                    |
| C2 | λ​\-RLM\\lambda\\text{-}\\textsf{RLM} (8B) vs RLM (70B)                                     | 35.7 vs 36.1 | 57 vs 177 | 8B ties 70B, 3.1×\\times faster                                 |
| C3 | λ​\-RLM\\lambda\\text{-}\\textsf{RLM} (8B) vs Direct (405B)                                 | 35.7 vs 27.2 | 57 vs 45  | 8B+λ\\lambda beats 405B accuracy                                |
| C4 | λ​\-RLM\\lambda\\text{-}\\textsf{RLM} (7B) vs λ​\-RLM\\lambda\\text{-}\\textsf{RLM} (Cdsrl) | 28.7 vs 44.6 | 65 vs 44  | Coding helps, gap = 16 pp                                       |
| C5 | RLM (405B) vs λ​\-RLM\\lambda\\text{-}\\textsf{RLM} (405B)                                  | 55.3 vs 57.9 | 112 vs 35 | λ​\-RLM\\lambda\\text{-}\\textsf{RLM} wins avg; RLM wins CodeQA |

#### C1: Formal structure helps weak models dramatically.

On the same Qwen3-8B model, replacing the ad-hoc REPL loop with pre-verified combinators yields +21.9+21.9 pp in accuracy and 4.0×4.0\\times latency reduction. This is the core contribution: the scaffold absorbs complexity that the weak model cannot handle.

#### C2: An 8B model with λ​\-RLM\\lambda\\text{-}\\textsf{RLM} matches a 70B model with Normal RLM.

Qwen3-8B under λ​\-RLM\\lambda\\text{-}\\textsf{RLM} achieves 35.7% average accuracy, statistically tied with Llama-70B under Normal RLM at 36.1%, while being 3.1×3.1\\times faster. This confirms that formal structure can substitute for raw model scale on long-context tasks.

#### C3: λ​\-RLM\\lambda\\text{-}\\textsf{RLM}(8B) outperforms Direct(405B) on accuracy.

Even the largest model, when fed the entire prompt directly, suffers from context rot. The 8B model with λ​\-RLM\\lambda\\text{-}\\textsf{RLM} achieves 35.7% vs 27.2% for Direct Llama-405B, though the direct call is faster (no scaffold overhead). This validates Corollary [5](#Thmtheorem5 "Corollary 5 (Scaling Laws). ‣ 4 Theoretical Guarantees ‣ The 𝐘-Combinator for LLMs: Solving Long-Context Rot with 𝜆-Calculus"): λ​\-RLM\\lambda\\text{-}\\textsf{RLM}’s power-law accuracy decay dominates the exponential decay of direct calls at long contexts.

#### C4: Coding ability still matters, but the gap narrows.

Mistral-7B (low code skill) under λ​\-RLM\\lambda\\text{-}\\textsf{RLM} achieves 28.7%, while Codestral-22B (high code skill) achieves 44.6% i.e a 16 pp gap. Under Normal RLM, the same pair shows a 28 pp gap (9.7% vs 37.5%). The combinator library reduces but does not eliminate the benefit of coding ability, because the leaf sub-prompts still benefit from the model’s language understanding quality.

#### C5: At the frontier, a nuanced tradeoff emerges.

On average, λ​\-RLM\\lambda\\text{-}\\textsf{RLM} (405B) narrowly outperforms RLM(405B) (57.9% vs 55.3%) while being 3.2×3.2\\times faster. However, on CodeQA specifically, RLM (405B) wins 62.1% vs 55.7%. This suggests that the fixed combinator library, while broadly superior, may benefit from task-specific extensions for code understanding.

#### Further Ablations.

To isolate the contribution of each λ​\-RLM\\lambda\\text{-}\\textsf{RLM} component, we run ablations on Qwen3-8B ×\\times OOLONG (Table [10](#S5.T10 "Table 10 ‣ Further Ablations. ‣ 5.1 Main Results ‣ 5 Experiments ‣ The 𝐘-Combinator for LLMs: Solving Long-Context Rot with 𝜆-Calculus")). We note that replacing the combinator library with free-form code generation drops accuracy by 24.2 pp and increases latency by 3.9×3.9\\times, which is exactly the Normal RLM result. The combinator library is the single largest contributor to λ​\-RLM\\lambda\\text{-}\\textsf{RLM}’s advantage.

Table 10: Ablation study on Qwen3-8B ×\\times OOLONG (O​(n)O(n) task, 131K tokens).

| ID | Ablation                                   | Acc (%) | Lat (s) | Δ\\DeltaAcc | Interpretation                  |
| -- | ------------------------------------------ | ------- | ------- | ----------- | ------------------------------- |
| \- | Full λ​\-RLM\\lambda\\text{-}\\textsf{RLM} | 48.3    | 62.4    | \-          | Full system                     |
| A1 | Random k∈\[2,100\]k\\in\[2,100\]           | 31.5    | 88.7    | −\-16.8     | Planner’s k∗k^{\*} matters      |
| A2 | Fixed task = “classify”                    | 41.2    | 65.1    | −\-7.1      | Task detection helps            |
| A3 | ⊕\=ℳ\\oplus=\\mathcal{M} (neural compose)  | 43.6    | 108.3   | −\-4.7      | Symbolic ⊕\\oplus saves latency |
| A4 | LLM writes free-form code                  | 24.1    | 241.6   | −\-24.2     | Combinator library is critical  |
| A5 | No pre-filter (process all)                | 46.8    | 74.2    | −\-1.5      | Pre-filter helps modestly       |

Furthermore, random chunk sizes lose 16.8 pp, validating Theorem [4](#Thmtheorem4 "Theorem 4 (Optimal Partition). ‣ 4 Theoretical Guarantees ‣ The 𝐘-Combinator for LLMs: Solving Long-Context Rot with 𝜆-Calculus"): the closed-form k∗k^{\*} provides a meaningful optimum. Qualitatively, random selection sometimes yields k\=2k=2 (too few chunks, large context rot) or k\=100k=100 (too many sub-calls, excessive overhead). Finally, replacing symbolic ⊕\=MergeCounts\\oplus=\\textsc{MergeCounts} with ℳ\\mathcal{M}\-based composition costs only 4.7 pp in accuracy but nearly doubles latency (62→10862\\to 108s), because every recursion level now requires an additional LLM call. This is the C⊕​(k∗)C\_{\\oplus}(k^{\*}) term from Theorem [2](#Thmtheorem2 "Theorem 2 (Cost Bound). ‣ 4 Theoretical Guarantees ‣ The 𝐘-Combinator for LLMs: Solving Long-Context Rot with 𝜆-Calculus"): when ⊕\\oplus is symbolic, C⊕\=0C\_{\\oplus}=0 and the cost recurrence simplifies to pure leaf cost.

## 6 Related Work

Our work sits at the intersection of long-context reasoning and formal methods. While recent efforts have focused on extending the native context window of Transformers, or delegating search to model-authored code, we argue that the primary bottleneck is not just memory size, but the lack of verifiable control flow during evidence aggregation.

#### Long-Context Scaling & Context Management.

While LLMs have become sophisticated general-purpose reasoners, they struggle with inputs that exceed these native limits, such as large codebases or multi-file repositories. Standard approaches to this problem often rely on simple heuristics. For example, naive truncation or sliding-window prompting are frequently used but often force the model to "forget" early information, breaking tasks that require systematic evidence gathering or global consistency (Dai et al., [2019](#bib.bib10); Liu et al., [2023](#bib.bib22); An et al., [2024](#bib.bib2); Bertsch et al., [2025](#bib.bib3); Fountas et al., [2025](#bib.bib12)).

Recent reframing of this problem focuses on inference-time scaling and decoding (Zimmer et al., [2025a](#bib.bib44); Chen et al., [2025](#bib.bib7); Lin et al., [2026](#bib.bib21); Ji et al., [2026b](#bib.bib18), [a](#bib.bib17)), where computation is scaled by decomposing problems into smaller subproblems (Xu et al., [2026a](#bib.bib33); Chen et al., [2026](#bib.bib8)). While retrieval-augmented generation (RAG) and architectural extensions (Jin et al., [2024](#bib.bib19); Li et al., [2024](#bib.bib20)) have been prominent, they often struggle with tasks requiring a holistic view of the input. Recent work in neuroSymbolic augmented reasoning (Nezhad and Agrawal, [2025](#bib.bib24); Hakim et al., [2025](#bib.bib16); Yang et al., [2025a](#bib.bib35)) has validated that structured reasoning layers can maintain performance in large contexts, where purely neural models typically fail.

#### Recursive & Hierarchical Reasoning.

λ​\-RLM\\lambda\\text{-}\\textsf{RLM} follows a lineage of hierarchical prompting strategies, such as "Least-to-Most" (Zhou et al., [2023](#bib.bib42)) prompting and "Tree-of-Thoughts" (Yao et al., [2023a](#bib.bib37)). The most direct predecessor is the RLM (Zhang et al., [2026](#bib.bib40)), which introduced the "prompt-as-environment" paradigm. However, standard RLMs rely on an open-ended REPL loop where the model generates arbitrary Python code to control its own recursion. This "stochastic control" introduces failure modes where code may not parse, recursion may run away, or execution becomes unpredictable. More recent follow-up work (Alizadeh et al., [2026](#bib.bib1)) improves this paradigm by using uncertainty-aware self-reflective program search to better select interaction programs under a fixed inference budget. In contrast, λ​\-RLM\\lambda\\text{-}\\textsf{RLM} addresses the reliability bottleneck from a different angle: instead of improving search over free-form control programs, it replaces model-authored control with a fixed library of deterministic combinators, shifting the model from an unconstrained controller to a bounded oracle for leaf-level subproblems.

#### Agentic Programming & Structured Control Flows.

Our framework situates itself within the rapid evolution of agentic programming. This paradigm shifts away from one-shot prompting toward iterative systems where LLMs autonomously plan and execute multi-step tasks (Mower et al., [2024](#bib.bib23); Grosnit et al., [2025](#bib.bib14); Sun et al., [2025](#bib.bib28)). However, the primary challenge in this field remains the "reliability gap": as agents gain more autonomy, their execution traces become increasingly difficult to audit or bound. Recent frameworks, such as control flows (Niu et al., [2025](#bib.bib25); Choi et al., [2025](#bib.bib9); Shi et al., [2025](#bib.bib27); Yu et al., [2025](#bib.bib39); Wang et al., [2025](#bib.bib31)), have attempted to mitigate this by allowing developers to define discrete, observable tasks for AI agents. Despite these efforts, many agentic systems still rely on the model to dynamically generate their own control flow. This introduces significant failure modes, including non-termination and malformed execution paths that are orthogonal to the underlying reasoning task (Zhu et al., [2025](#bib.bib43); Cemri et al., [2025](#bib.bib6); Zhang et al., [2025](#bib.bib41)). Importantly, the risks of open-ended control are not merely operational but also structural. Recent research into memory control flow attacks (Xu et al., [2026b](#bib.bib34)) demonstrates that manipulating an agent’s tool-call or memory trace can induce catastrophic reasoning failures (Vogelsang, [2024](#bib.bib29); Wu et al., [2024](#bib.bib32); Guo et al., [2024](#bib.bib15)).

λ​\-RLMs\\lambda\\text{-}\\textsf{RLMs} address these vulnerabilities by strictly separating reasoning content (which remains neural) from control flow (which becomes symbolic and deterministic). By enforcing a restricted functional runtime, we provide formal guarantees that are currently absent from general-purpose agentic scaffolds.

#### Neuro-Symbolic Integration & Formal Methods.

The use of λ\\lambda\-calculus to manage LLM control flow represents a deep integration of neural inference and symbolic logic. This aligns with recent theoretical work, such as (Dong et al., [2024](#bib.bib11); Oldenburg, [2023](#bib.bib26); Garby et al., [2026](#bib.bib13); Bhardwaj, [2026](#bib.bib4)). Specifically, λ​\-RLM\\lambda\\text{-}\\textsf{RLM} utilises fixed-point combinators (e.g., the Y-combinator) to express recursion as a first-class semantic object rather than an emergent side effect of model prompting. This formal grounding allows us to prove properties that remain absent from standard, non-typed recursive models.

## 7 Conclusions and Future Work

In this paper, we introduced λ​\-RLMs\\lambda\\text{-}\\textsf{RLMs}, a framework that reframes long-context reasoning as a structured functional program grounded in λ\\lambda\-calculus. By replacing the open-ended REPL loops of standard recursive language models with a typed runtime of deterministic combinators, we effectively separated neural reasoning from symbolic control. This architectural shift addresses the primary failure modes of existing scaffolds: unpredictability, non-termination, and the high "coding tax" imposed on smaller models. Our empirical evaluation across nine model tiers and four complex tasks demonstrates that formal structure is a powerful substitute for parameter scale. Most notably, we showed that a properly scaffolded 8B model can match or exceed the accuracy of a 70B model using standard recursive methods while delivering up to 4.1×4.1\\times reductions in latency. Beyond performance, λ​\-RLMs\\lambda\\text{-}\\textsf{RLMs} provide a level of mathematical rigour previously absent from this domain, including guaranteed termination and closed-form cost bounds.

The success of λ​\-RLMs\\lambda\\text{-}\\textsf{RLMs} suggests that the future of reliable AI lies not in giving models unrestricted freedom to program their own execution, but in providing them with high-integrity, verifiable environments. While this work focused on the immediate bottleneck of long-context reasoning, the underlying principle, treating the LLM as a bounded oracle within a formal functional structure, has broader implications for the design of intelligent systems.

## References

* Alizadeh et al. \[2026\] Keivan Alizadeh, Parshin Shojaee, Minsik Cho, and Mehrdad Farajtabar. Recursive language models meet uncertainty: The surprising effectiveness of self-reflective program search for long context, 2026. URL <https://arxiv.org/abs/2603.15653>.
* An et al. \[2024\] Chenxin An, Fei Huang, Jun Zhang, Shansan Gong, Xipeng Qiu, Chang Zhou, and Lingpeng Kong. Training-free long-context scaling of large language models. _arXiv preprint arXiv:2402.17463_, 2024.
* Bertsch et al. \[2025\] Amanda Bertsch, Maor Ivgi, Emily Xiao, Uri Alon, Jonathan Berant, Matthew R Gormley, and Graham Neubig. In-context learning with long-context models: An in-depth exploration. In _Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)_, pages 12119–12149, 2025.
* Bhardwaj \[2026\] Varun Pratap Bhardwaj. Formal analysis and supply chain security for agentic ai skills. _arXiv preprint arXiv:2603.00195_, 2026.
* Brown et al. \[2020\] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners, 2020. URL <https://arxiv.org/abs/2005.14165>.
* Cemri et al. \[2025\] Mert Cemri, Melissa Z. Pan, Shuyi Yang, Lakshya A. Agrawal, Bhavya Chopra, Rishabh Tiwari, Kurt Keutzer, Aditya Parameswaran, Dan Klein, Kannan Ramchandran, Matei Zaharia, Joseph E. Gonzalez, and Ion Stoica. Why do multi-agent llm systems fail?, 2025. URL <https://arxiv.org/abs/2503.13657>.
* Chen et al. \[2025\] Guanzheng Chen, Qilong Feng, Jinjie Ni, Xin Li, and Michael Qizhe Shieh. Rapid: Long-context inference with retrieval-augmented speculative decoding, 2025. URL <https://arxiv.org/abs/2502.20330>.
* Chen et al. \[2026\] Shengkai Chen, Zhiguang Cao, Jianan Zhou, Yaoxin Wu, Senthilnath Jayavelu, Zhuoyi Lin, Xiaoli Li, and Shili Xiang. Dragon: Llm-driven decomposition and reconstruction agents for large-scale combinatorial optimization, 2026. URL <https://arxiv.org/abs/2601.06502>.
* Choi et al. \[2025\] Jae-Woo Choi, Hyungmin Kim, Hyobin Ong, Minsu Jang, Dohyung Kim, Jaehong Kim, and Youngwoo Yoon. Reactree: Hierarchical llm agent trees with control flow for long-horizon task planning. _arXiv preprint arXiv:2511.02424_, 2025.
* Dai et al. \[2019\] Zihang Dai, Zhilin Yang, Yiming Yang, Jaime Carbonell, Quoc V. Le, and Ruslan Salakhutdinov. Transformer-xl: Attentive language models beyond a fixed-length context, 2019. URL <https://arxiv.org/abs/1901.02860>.
* Dong et al. \[2024\] Liming Dong, Qinghua Lu, and Liming Zhu. Agentops: Enabling observability of llm agents. _arXiv preprint arXiv:2411.05285_, 2024.
* Fountas et al. \[2025\] Zafeirios Fountas, Martin A Benfeghoul, Adnan Oomerjee, Fenia Christopoulou, Gerasimos Lampouras, Haitham Bou-Ammar, and Jun Wang. Human-inspired episodic memory for infinite context llms, 2025. URL <https://arxiv.org/abs/2407.09450>.
* Garby et al. \[2026\] Zac Garby, Andrew D Gordon, and David Sands. The llmbda calculus: Ai agents, conversations, and information flow. _arXiv preprint arXiv:2602.20064_, 2026.
* Grosnit et al. \[2025\] Antoine Grosnit, Alexandre Maraval, Refinath S N, Zichao Zhao, James Doran, Giuseppe Paolo, Albert Thomas, Jonas Gonzalez, Abhineet Kumar, Khyati Khandelwal, Abdelhakim Benechehab, Hamza Cherkaoui, Youssef Attia El-Hili, Kun Shao, Jianye Hao, Jun Yao, Balázs Kégl, Haitham Bou-Ammar, and Jun Wang. Kolb-based experiential learning for generalist agents with human-level kaggle data science performance, 2025. URL <https://arxiv.org/abs/2411.03562>.
* Guo et al. \[2024\] Xingang Guo, Fangxu Yu, Huan Zhang, Lianhui Qin, and Bin Hu. Cold-attack: Jailbreaking llms with stealthiness and controllability, 2024. URL <https://arxiv.org/abs/2402.08679>.
* Hakim et al. \[2025\] Safayat Bin Hakim, Muhammad Adil, Alvaro Velasquez, and Houbing Herbert Song. Symrag: Efficient neuro-symbolic retrieval through adaptive query routing, 2025. URL <https://arxiv.org/abs/2506.12981>.
* Ji et al. \[2026a\] Xiaotong Ji, Rasul Tutunov, Matthieu Zimmer, and Haitham Bou Ammar. Scalable power sampling: Unlocking efficient, training-free reasoning for llms via distribution sharpening, 2026a. URL <https://arxiv.org/abs/2601.21590>.
* Ji et al. \[2026b\] Xiaotong Ji, Rasul Tutunov, Matthieu Zimmer, and Haitham Bou-Ammar. Decoding as optimisation on the probability simplex: From top-k to top-p (nucleus) to best-of-k samplers, 2026b. URL <https://arxiv.org/abs/2602.18292>.
* Jin et al. \[2024\] Bowen Jin, Jinsung Yoon, Jiawei Han, and Sercan O. Arik. Long-context llms meet rag: Overcoming challenges for long inputs in rag, 2024. URL <https://arxiv.org/abs/2410.05983>.
* Li et al. \[2024\] Zhuowan Li, Cheng Li, Mingyang Zhang, Qiaozhu Mei, and Michael Bendersky. Retrieval augmented generation or long-context llms? a comprehensive study and hybrid approach, 2024. URL <https://arxiv.org/abs/2407.16833>.
* Lin et al. \[2026\] Gang Lin, Dongfang Li, Zhuoen Chen, Yukun Shi, Xuhui Chen, Baotian Hu, and Min Zhang. Lycheedecode: Accelerating long-context llm inference via hybrid-head sparse decoding, 2026. URL <https://arxiv.org/abs/2602.04541>.
* Liu et al. \[2023\] Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. Lost in the middle: How language models use long contexts, 2023. URL <https://arxiv.org/abs/2307.03172>.
* Mower et al. \[2024\] Christopher E. Mower, Yuhui Wan, Hongzhan Yu, Antoine Grosnit, Jonas Gonzalez-Billandon, Matthieu Zimmer, Jinlong Wang, Xinyu Zhang, Yao Zhao, Anbang Zhai, Puze Liu, Daniel Palenicek, Davide Tateo, Cesar Cadena, Marco Hutter, Jan Peters, Guangjian Tian, Yuzheng Zhuang, Kun Shao, Xingyue Quan, Jianye Hao, Jun Wang, and Haitham Bou-Ammar. Ros-llm: A ros framework for embodied ai with task feedback and structured reasoning, 2024. URL <https://arxiv.org/abs/2406.19741>.
* Nezhad and Agrawal \[2025\] Sina Bagheri Nezhad and Ameeta Agrawal. Enhancing large language models with neurosymbolic reasoning for multilingual tasks, 2025. URL <https://arxiv.org/abs/2506.02483>.
* Niu et al. \[2025\] Boye Niu, Yiliao Song, Kai Lian, Yifan Shen, Yu Yao, Kun Zhang, and Tongliang Liu. Flow: Modularized agentic workflow automation, 2025. URL <https://arxiv.org/abs/2501.07834>.
* Oldenburg \[2023\] Reinhard Oldenburg. Limitations of and lessons from the learning of large language models. 2023.
* Shi et al. \[2025\] Yuchen Shi, Siqi Cai, Zihan Xu, Yulei Qin, Gang Li, Hang Shao, Jiawei Chen, Deqing Yang, Ke Li, and Xing Sun. Flowagent: a new paradigm for workflow agent. 2025.
* Sun et al. \[2025\] Weiwei Sun, Miao Lu, Zhan Ling, Kang Liu, Xuesong Yao, Yiming Yang, and Jiecao Chen. Scaling long-horizon llm agent via context-folding. _arXiv preprint arXiv:2510.11967_, 2025.
* Vogelsang \[2024\] Terry Vogelsang. Llm controls execution flow hijacking. In _Large Language Models in Cybersecurity: Threats, Exposure and Mitigation_, pages 99–104\. Springer, 2024.
* Wang et al. \[2024\] Xindi Wang, Mahsa Salmani, Parsa Omidi, Xiangyu Ren, Mehdi Rezagholizadeh, and Armaghan Eshaghi. Beyond the limits: A survey of techniques to extend the context length in large language models, 2024. URL <https://arxiv.org/abs/2402.02244>.
* Wang et al. \[2025\] Zhaodong Wang, Samuel Lin, Guanqing Yan, Soudeh Ghorbani, Minlan Yu, Jiawei Zhou, Nathan Hu, Lopa Baruah, Sam Peters, Srikanth Kamath, et al. Intent-driven network management with multi-agent llms: The confucius framework. In _Proceedings of the ACM SIGCOMM 2025 Conference_, pages 347–362, 2025.
* Wu et al. \[2024\] Fangzhou Wu, Ethan Cecchetti, and Chaowei Xiao. System-level defense against indirect prompt injection attacks: An information flow control perspective. _arXiv preprint arXiv:2409.19091_, 2024.
* Xu et al. \[2026a\] Zhen Xu, Shang Zhu, Jue Wang, Junlin Wang, Ben Athiwaratkun, Chi Wang, James Zou, and Ce Zhang. When does divide and conquer work for long context llm? a noise decomposition framework, 2026a. URL <https://arxiv.org/abs/2506.16411>.
* Xu et al. \[2026b\] Zhenlin Xu, Xiaogang Zhu, Yu Yao, Minhui Xue, and Yiliao Song. From storage to steering: Memory control flow attacks on llm agents, 2026b. URL <https://arxiv.org/abs/2603.15125>.
* Yang et al. \[2025a\] Xiao-Wen Yang, Jie-Jing Shao, Lan-Zhe Guo, Bo-Wen Zhang, Zhi Zhou, Lin-Han Jia, Wang-Zhou Dai, and Yu-Feng Li. Neuro-symbolic artificial intelligence: Towards improving the reasoning abilities of large language models, 2025a. URL <https://arxiv.org/abs/2508.13678>.
* Yang et al. \[2025b\] Zhuoyi Yang, Xu Guo, Tong Zhang, Huijuan Xu, and Boyang Li. Test-time scaling of llms: A survey from a subproblem structure perspective, 2025b. URL <https://arxiv.org/abs/2511.14772>.
* Yao et al. \[2023a\] Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models, 2023a. URL <https://arxiv.org/abs/2305.10601>.
* Yao et al. \[2023b\] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models, 2023b. URL <https://arxiv.org/abs/2210.03629>.
* Yu et al. \[2025\] Chaojia Yu, Zihan Cheng, Hanwen Cui, Yishuo Gao, Zexu Luo, Yijin Wang, Hangbin Zheng, and Yong Zhao. A survey on agent workflow–status and future. In _2025 8th International Conference on Artificial Intelligence and Big Data (ICAIBD)_, pages 770–781\. IEEE, 2025.
* Zhang et al. \[2026\] Alex L. Zhang, Tim Kraska, and Omar Khattab. Recursive language models, 2026. URL <https://arxiv.org/abs/2512.24601>.
* Zhang et al. \[2025\] Shaokun Zhang, Ming Yin, Jieyu Zhang, Jiale Liu, Zhiguang Han, Jingyang Zhang, Beibin Li, Chi Wang, Huazheng Wang, Yiran Chen, and Qingyun Wu. Which agent causes task failures and when? on automated failure attribution of llm multi-agent systems, 2025. URL <https://arxiv.org/abs/2505.00212>.
* Zhou et al. \[2023\] Denny Zhou, Nathanael Schärli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schuurmans, Claire Cui, Olivier Bousquet, Quoc Le, and Ed Chi. Least-to-most prompting enables complex reasoning in large language models, 2023. URL <https://arxiv.org/abs/2205.10625>.
* Zhu et al. \[2025\] Kunlun Zhu, Zijia Liu, Bingxuan Li, Muxin Tian, Yingxuan Yang, Jiaxun Zhang, Pengrui Han, Qipeng Xie, Fuyang Cui, Weijia Zhang, Xiaoteng Ma, Xiaodong Yu, Gowtham Ramesh, Jialian Wu, Zicheng Liu, Pan Lu, James Zou, and Jiaxuan You. Where llm agents fail and how they can learn from failures, 2025. URL <https://arxiv.org/abs/2509.25370>.
* Zimmer et al. \[2025a\] Matthieu Zimmer, Milan Gritta, Gerasimos Lampouras, Haitham Bou Ammar, and Jun Wang. Mixture of attentions for speculative decoding, 2025a. URL <https://arxiv.org/abs/2410.03804>.
* Zimmer et al. \[2025b\] Matthieu Zimmer, Xiaotong Ji, Rasul Tutunov, Anthony Bordg, Jun Wang, and Haitham Bou Ammar. Bourbaki: Self-generated and goal-conditioned mdps for theorem proving, 2025b. URL <https://arxiv.org/abs/2507.02726>.

## Appendix A Complete Example Trace

We trace the λ​\-RLM\\lambda\\text{-}\\textsf{RLM} on the OOLONG task: classifying 1000 questions in 131K tokens, with K\=32K=32K.

Phase 1 \- DetectTask: τtype\=aggregate\\tau\_{\\text{type}}=\\texttt{aggregate} \[1 LLM call\] Phase 2 \- Plan: k∗\=5,τ∗\=26K,⊕\=MergeCounts,d\=1k^{\*}=5,\\ \\tau^{\*}=26\\text{K},\\ \\oplus=\\textsc{MergeCounts},\\ d=1 \[0 LLM calls\] Phase 3 \- EstimateCost: C^\=5×$​0.03+$​0.02\=$​0.17\\hat{C}=5\\times\\mathdollar 0.03+\\mathdollar 0.02=\\mathdollar 0.17, N^\=6\\hat{N}=6 calls \[0 LLM calls\] Phase 4 \- Execute: \[5 LLM calls\] Split​(P,5)→\[P1,P2,P3,P4,P5\]\\displaystyle\\textsc{Split}(P,5)\\to\[P\_{1},P\_{2},P\_{3},P\_{4},P\_{5}\] symbolic: free Map(λpi.ℳ(“count categories: ”∥pi),\[P1:5\])\\displaystyle\\textsc{Map}(\\lambda p\_{i}.\\,\\mathcal{M}(\\text{\`\`count categories: ''}\\|p\_{i}),\[P\_{1:5}\]) neural: 5 β\\beta\-reductions →\[{desc:45,num:52,…},…,{desc:33,num:42,…}\]\\displaystyle\\quad\\to\[\\{\\text{desc}:45,\\text{num}:52,\\ldots\\},\\ldots,\\{\\text{desc}:33,\\text{num}:42,\\ldots\\}\] Reduce​(MergeCounts,results)→{desc:200,num:240,…}\\displaystyle\\textsc{Reduce}(\\textsc{MergeCounts},\\text{results})\\to\\{\\text{desc}:200,\\text{num}:240,\\ldots\\} symbolic: free Answer: “description is less common than numeric value” ✓\\displaystyle\\checkmark Total: 6 LLM calls, $0.17, correct. Normal RLM on same task: Huge no of LLM calls, $1.12, incorrect (Example E.2 in \[Zhang et al., [2026](#bib.bib40)\]) 

## Appendix B The Hierarchy of Computation

The λ​\-RLM\\lambda\\text{-}\\textsf{RLM} cleanly separates computation into three layers:

Layer 1: Symbolic (Lambda Calculus)Split, Map, Filter, Reduce, Cross, Concat, PeekDeterministic ⋅\\cdot Pre-verified ⋅\\cdot Zero cost ⋅\\cdot Guaranteed terminationLayer 2: Planning (Optimization)k∗\=⌈n⋅cin/c⊕⌉k^{\*}=\\lceil\\sqrt{n\\cdot c\_{\\text{in}}/c\_{\\oplus}}\\rceil, τ∗\=min⁡(K,n/k∗)\\tau^{\*}=\\min(K,n/k^{\*}), depth \=⌈logk⁡(n/K)⌉\=\\lceil\\log\_{k}(n/K)\\rceilPre-computed ⋅\\cdot Deterministic cost ⋅\\cdot Accuracy-constrainedLayer 3: Neural (β\\beta\-Reductions at Leaves)ℳ​(Pi)\\mathcal{M}(P\_{i}) where |Pi|≤τ∗≤K|P\_{i}|\\leq\\tau^{\*}\\leq KThe ONLY uncertain component ⋅\\cdot Each call within context windowparametersleaf calls 

## Appendix C Proofs

See [1](#Thmtheorem1 "Theorem 1 (Termination). ‣ 4 Theoretical Guarantees ‣ The 𝐘-Combinator for LLMs: Solving Long-Context Rot with 𝜆-Calculus") 

###### Proof.

Define the _rank_ of a call Φ​(P′)\\Phi(P^{\\prime}) as r​(P′)\=⌈logk∗⁡(|P′|/τ∗)⌉r(P^{\\prime})=\\lceil\\log\_{k^{\*}}(|P^{\\prime}|/\\tau^{\*})\\rceil. We prove termination by strong induction on rank.

Base case (r\=0r=0): |P′|≤τ∗≤K|P^{\\prime}|\\leq\\tau^{\*}\\leq K, so Φ\\Phi invokes sub\_​ℳ​(q)\\texttt{sub\\\_}\\mathcal{M}(q) via the REPL, which halts by (A1).

Inductive step: Suppose Φ\\Phi terminates for all inputs with rank <r<r.

Now, if |P′||P^{\\prime}| has rank r\>0r>0,them r−1≤logk⋆⁡|P|′τ∗≤rr-1\\leq\\log\_{k^{\\star}}\\frac{|P|^{\\prime}}{\\tau^{\*}}\\leq r. Hence the rank rr is an integer and r\>0r>0 implies r≥1r\\geq 1, then |P′|\>τ∗|P^{\\prime}|>\\tau^{\*} and Φ\\Phi executes Split​(P′,k∗)\\textsc{Split}(P^{\\prime},k^{\*}) in the REPL, which halts by (A2), producing chunks P1,…,Pk∗P\_{1},\\ldots,P\_{k^{\*}} with |Pi|\=⌈|P′|/k∗⌉|P\_{i}|=\\lceil|P^{\\prime}|/k^{\*}\\rceil. Since

| \|Pi|\=⌈|P′|k∗⌉<|P′|(as ​k∗≥2),|P\_{i}|=\\left\\lceil\\frac{|P^{\\prime}|}{k^{\*}}\\right\\rceil<|P^{\\prime}|\\quad(\\text{as }k^{\*}\\geq 2), | (9) |
| ----------------------------------------------------------------------------------------------------------------------------------------------- | --- |

we have r​(Pi)<r​(P′)r(P\_{i})<r(P^{\\prime}), so each recursive call Φ​(Pi)\\Phi(P\_{i}) terminates by the inductive hypothesis. The REPL-executed operations Map, Reduce, and Filter all halt by (A2). Thus Φ​(P′)\\Phi(P^{\\prime}) terminates.

For the call count: at depth ℓ∈{0,…,d−1}\\ell\\in\\{0,\\ldots,d\\!-\\!1\\}, there are (k∗)ℓ(k^{\*})^{\\ell} nodes each spawning k∗k^{\*} children. The leaves reside at depth dd, giving (k∗)d(k^{\*})^{d} leaf calls to sub\_​ℳ\\texttt{sub\\\_}\\mathcal{M}. Adding the single task-detection call (line 6 of Algorithm [1](#alg1 "Algorithm 1 ‣ 3.2 Core Formulation ‣ 3 The 𝜆⁢"-RLM" Framework ‣ The 𝐘-Combinator for LLMs: Solving Long-Context Rot with 𝜆-Calculus")) gives N​(n)\=(k∗)d+1N(n)=(k^{\*})^{d}+1.

Contrast with original RLM. The loop ([3](#S3.E3 "In 3.1 From Open-Ended Control to a Restricted Runtime ‣ 3 The 𝜆⁢"-RLM" Framework ‣ The 𝐘-Combinator for LLMs: Solving Long-Context Rot with 𝜆-Calculus")) has no rank function; the LLM may generate code that does not reduce input size, yielding unbounded iterations. λ​\-RLM\\lambda\\text{-}\\textsf{RLM} eliminates this by construction: every recursive call strictly reduces rank. ∎

See [2](#Thmtheorem2 "Theorem 2 (Cost Bound). ‣ 4 Theoretical Guarantees ‣ The 𝐘-Combinator for LLMs: Solving Long-Context Rot with 𝜆-Calculus") 

###### Proof.

To start, we unroll the recurrence in Equation ([6](#S4.E6 "In Theorem 2 (Cost Bound). ‣ 4 Theoretical Guarantees ‣ The 𝐘-Combinator for LLMs: Solving Long-Context Rot with 𝜆-Calculus")). At level ℓ\\ell, there are (k∗)ℓ(k^{\*})^{\\ell} subproblems each of size n/(k∗)ℓn/(k^{\*})^{\\ell}, and one composition step costing 𝒞⊕​(k∗)\\mathcal{C}\_{\\oplus}(k^{\*}). Expanding:

| T​(n)\\displaystyle T(n)                                                                                                                                                                                       | \=k∗⋅T​(n/k∗)+𝒞⊕​(k∗)\=k∗​\[k∗⋅T​(n/(k∗)2)+𝒞⊕​(k∗)\]+𝒞⊕​(k∗)\\displaystyle=k^{\*}\\cdot T(n/k^{\*})+\\mathcal{C}\_{\\oplus}(k^{\*})=k^{\*}\\bigl\[k^{\*}\\cdot T(n/(k^{\*})^{2})+\\mathcal{C}\_{\\oplus}(k^{\*})\\bigr\]+\\mathcal{C}\_{\\oplus}(k^{\*}) |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| \=(k∗)2⋅T​(n(k∗)2)+(k∗+1)⋅𝒞⊕​(k∗)\\displaystyle=(k^{\*})^{2}\\cdot T\\!\\left(\\frac{n}{(k^{\*})^{2}}\\right)+(k^{\*}+1)\\cdot\\mathcal{C}\_{\\oplus}(k^{\*})\\                                               |                                                                                                                                                                                                                                                             |
| \=(k∗)d⋅T​(n(k∗)d)+((k∗)d−1+…+k∗+1)⋅𝒞⊕​(k∗)\\displaystyle=(k^{\*})^{d}\\cdot T\\!\\left(\\frac{n}{(k^{\*})^{d}}\\right)+((k^{\*})^{d-1}+\\ldots+k^{\*}+1)\\cdot\\mathcal{C}\_{\\oplus}(k^{\*})                |                                                                                                                                                                                                                                                             |
| \=(k∗)d⋅T​(n(k∗)d)+\[(k∗)d−1k∗−1\]⋅𝒞⊕​(k∗)\\displaystyle=(k^{\*})^{d}\\cdot T\\!\\left(\\frac{n}{(k^{\*})^{d}}\\right)+\\left\[\\frac{(k^{\*})^{d}-1}{k^{\*}-1}\\right\]\\cdot\\mathcal{C}\_{\\oplus}(k^{\*}) |                                                                                                                                                                                                                                                             |

At depth d\=⌈logk∗⁡(n/τ∗)⌉d=\\lceil\\log\_{k^{\*}}(n/\\tau^{\*})\\rceil, we have d−1≤logk∗⁡nτ∗≤dd-1\\leq\\log\_{k^{\*}}\\frac{n}{\\tau^{\*}}\\leq d, hence (k∗)d≤n​k∗τ∗(k^{\*})^{d}\\leq\\frac{nk^{\*}}{\\tau^{\*}} and n(k∗)d≤τ∗\\frac{n}{(k^{\*})^{d}}\\leq\\tau^{\*}. These two results allow us to bound the above expression:

| T​(n)≤n​k∗τ∗​T​(τ∗)+\[n​k∗−τ∗τ∗​(k∗−1)\]⋅𝒞⊕​(k∗)\=n​k∗τ∗​𝒞​(τ∗)+\[n​k∗−τ∗τ∗​(k∗−1)\]⋅𝒞⊕​(k∗)\\displaystyle T(n)\\leq\\frac{nk^{\*}}{\\tau^{\*}}T(\\tau^{\*})+\\left\[\\frac{nk^{\*}-\\tau^{\*}}{\\tau^{\*}(k^{\*}-1)}\\right\]\\cdot\\mathcal{C}\_{\\oplus}(k^{\*})=\\frac{nk^{\*}}{\\tau^{\*}}\\mathcal{C}(\\tau^{\*})+\\left\[\\frac{nk^{\*}-\\tau^{\*}}{\\tau^{\*}(k^{\*}-1)}\\right\]\\cdot\\mathcal{C}\_{\\oplus}(k^{\*}) |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |

hitting the base case T​(τ∗)\=𝒞​(τ∗)T(\\tau^{\*})=\\mathcal{C}(\\tau^{\*}).

∎

See [3](#Thmtheorem3 "Theorem 3 (Accuracy Bound). ‣ 4 Theoretical Guarantees ‣ The 𝐘-Combinator for LLMs: Solving Long-Context Rot with 𝜆-Calculus") 

###### Proof.

The recursion tree of Φ\\Phi has depthd\=⌈logk∗⁡(n/τ∗)⌉d=\\lceil\\log\_{k^{\*}}(n/\\tau^{\*})\\rceil, branching factor k∗k^{\*}, and therefore (k∗)d(k^{\*})^{d} leaf nodes. By definition of the ceiling function:

| nτ∗≤(k∗)d≤n​k∗τ∗.\\frac{n}{\\tau^{\*}}\\;\\leq\\;(k^{\*})^{d}\\;\\leq\\;\\frac{n\\,k^{\*}}{\\tau^{\*}}. | (10) |
| ------------------------------------------------------------------------------------------------------- | ---- |

Base case (d\=0d=0): |P|≤τ∗|P|\\leq\\tau^{\*}. A single call to sub\_​ℳ\\texttt{sub\\\_}\\mathcal{M} yields accuracy 𝒜​(τ∗)\\mathcal{A}(\\tau^{\*}).

General case (d≥1d\\geq 1): Correctness requires (i) all (k∗)d(k^{\*})^{d} leaves correct, and (ii) all dd compositions correct, giving𝒜λ​\-RLM​(n)≥𝒜​(τ∗)(k∗)d⋅𝒜⊕d\\mathcal{A}\_{\\lambda\\text{-}\\textsf{RLM}}(n)\\geq\\mathcal{A}(\\tau^{\*})^{(k^{\*})^{d}}\\cdot\\mathcal{A}\_{\\oplus}^{d}.

Since 0<𝒜​(τ∗)<10<\\mathcal{A}(\\tau^{\*})<1, the mapx↦𝒜​(τ∗)xx\\mapsto\\mathcal{A}(\\tau^{\*})^{x} is strictly decreasing. Applying the upper bound from ([10](#A3.E10 "In Proof. ‣ Appendix C Proofs ‣ The 𝐘-Combinator for LLMs: Solving Long-Context Rot with 𝜆-Calculus")):

| 𝒜​(τ∗)(k∗)d≥𝒜​(τ∗)n​k∗/τ∗,\\mathcal{A}(\\tau^{\*})^{(k^{\*})^{d}}\\;\\geq\\;\\mathcal{A}(\\tau^{\*})^{nk^{\*}/\\tau^{\*}}, | (11) |
| ---------------------------------------------------------------------------------------------------------------------------- | ---- |

which yields the stated bound ([8](#S4.E8 "In item (ii) ‣ Theorem 3 (Accuracy Bound). ‣ 4 Theoretical Guarantees ‣ The 𝐘-Combinator for LLMs: Solving Long-Context Rot with 𝜆-Calculus")). ∎

See [4](#Thmtheorem4 "Theorem 4 (Optimal Partition). ‣ 4 Theoretical Guarantees ‣ The 𝐘-Combinator for LLMs: Solving Long-Context Rot with 𝜆-Calculus") 

###### Proof.

From ([7](#S4.E7 "In Theorem 2 (Cost Bound). ‣ 4 Theoretical Guarantees ‣ The 𝐘-Combinator for LLMs: Solving Long-Context Rot with 𝜆-Calculus")):

| T​(n,k)≤n​kτ∗​𝒞​(τ∗)+\[n​k−τ∗τ∗​(k−1)\]⋅𝒞⊕​(k)T(n,k)\\leq\\frac{nk}{\\tau^{\*}}\\mathcal{C}(\\tau^{\*})+\\left\[\\frac{nk-\\tau^{\*}}{\\tau^{\*}(k-1)}\\right\]\\cdot\\mathcal{C}\_{\\oplus}(k) |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |

or, after simplifying (treating τ∗\\tau^{\*} and nn as constants with respect to kk):

| T​(n,k)\\displaystyle T(n,k)                                                                                                                                          | ≤k​n⋅𝒞​(τ∗)τ∗⏟α+k2(k−1)​c⊕⋅nτ∗⏟β−k(k−1)​c⊕⏟γ\\displaystyle\\leq k\\underbrace{\\frac{n\\cdot\\mathcal{C}(\\tau^{\*})}{\\tau^{\*}}}\_{\\alpha}+\\frac{k^{2}}{(k-1)}\\underbrace{\\frac{c\_{\\oplus}\\cdot n}{\\tau^{\*}}}\_{\\beta}-\\frac{k}{(k-1)}\\underbrace{c\_{\\oplus}}\_{\\gamma} | (12) |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---- |
| \=α​k+β​k2k−1−γ​kk−1\\displaystyle=\\alpha k+\\beta\\frac{k^{2}}{k-1}-\\gamma\\frac{k}{k-1}                                                                           |                                                                                                                                                                                                                                                                                           |      |
| \=α​k​(k−1)+β​k2−γ​kk−1\=k2​(α+β)−k​(α+γ)k−1,\\displaystyle=\\frac{\\alpha k(k-1)+\\beta k^{2}-\\gamma k}{k-1}=\\frac{k^{2}(\\alpha+\\beta)-k(\\alpha+\\gamma)}{k-1}, |                                                                                                                                                                                                                                                                                           |      |

where nτ∗≤(k)d≤n​kτ∗\\frac{n}{\\tau^{\*}}\\leq(k)^{d}\\leq\\frac{nk}{\\tau^{\*}}. Taking the derivative to the upper-bound with respect to kk gives:

| dd​k​\[k2​(α+β)−k​(α+γ)k−1\]\\displaystyle\\frac{d}{dk}\\left\[\\frac{k^{2}(\\alpha+\\beta)-k(\\alpha+\\gamma)}{k-1}\\right\]                                                                                                                                                                | \=\[2​(α+β)​k−(α+γ)\]​(k−1)−\[k2​(α+β)−k​(α+γ)\](k−1)2\\displaystyle=\\frac{\\left\[2(\\alpha+\\beta)k-(\\alpha+\\gamma)\\right\](k-1)-\\left\[k^{2}(\\alpha+\\beta)-k(\\alpha+\\gamma)\\right\]}{(k-1)^{2}} |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| \=k2​(2​α+2​β−α−β)−2​(α+β)​k+α+γ(k−1)2\\displaystyle=\\frac{k^{2}(2\\alpha+2\\beta-\\alpha-\\beta)-2(\\alpha+\\beta)k+\\alpha+\\gamma}{(k-1)^{2}}                                                                                                                                            |                                                                                                                                                                                                              |
| \=k2​(α+β)−2​(α+β)​k+(α+γ)(k−1)2\\displaystyle=\\frac{k^{2}(\\alpha+\\beta)-2(\\alpha+\\beta)k+(\\alpha+\\gamma)}{(k-1)^{2}}                                                                                                                                                                 |                                                                                                                                                                                                              |
| \=(α+β)​\[(k−(1−1−α+γα+β))​(k−(1+1−α+γα+β))\](k−1)2\\displaystyle=\\frac{(\\alpha+\\beta)\\left\[\\left(k-\\left(1-\\sqrt{1-\\frac{\\alpha+\\gamma}{\\alpha+\\beta}}\\right)\\right)\\left(k-\\left(1+\\sqrt{1-\\frac{\\alpha+\\gamma}{\\alpha+\\beta}}\\right)\\right)\\right\]}{(k-1)^{2}} |                                                                                                                                                                                                              |

The minimum of the function is k∗k^{\*} closest to the value ⌈1+1−α+γα+β⌉\\lceil 1+\\sqrt{1-\\frac{\\alpha+\\gamma}{\\alpha+\\beta}}\\rceil. Because we have k≥2k\\geq 2, this implies that the optimal value of k∗\=2k^{\*}=2. ∎

## Appendix D Algorithmic Details

This section provides the algorithmic details of λ\\lambda\-RLM that complement the framework description in Section [3](#S3 "3 The 𝜆⁢"-RLM" Framework ‣ The 𝐘-Combinator for LLMs: Solving Long-Context Rot with 𝜆-Calculus"). In particular, we make explicit the end-to-end pipeline and the construction and execution of the fixed combinator program Φ\\Phi. We use a constant preview budget b\=500b=500 for inexpensive symbolic inspection. Algorithm [1](#alg1 "Algorithm 1 ‣ 3.2 Core Formulation ‣ 3 The 𝜆⁢"-RLM" Framework ‣ The 𝐘-Combinator for LLMs: Solving Long-Context Rot with 𝜆-Calculus") specifies the full end-to-end pipeline, Algorithm [4](#alg4 "Algorithm 4 ‣ Appendix D Algorithmic Details ‣ The 𝐘-Combinator for LLMs: Solving Long-Context Rot with 𝜆-Calculus") defines the recursive executor that carries out the planned decomposition, and Algorithms [5](#alg5 "Algorithm 5 ‣ Appendix D Algorithmic Details ‣ The 𝐘-Combinator for LLMs: Solving Long-Context Rot with 𝜆-Calculus") and [6](#alg6 "Algorithm 6 ‣ Appendix D Algorithmic Details ‣ The 𝐘-Combinator for LLMs: Solving Long-Context Rot with 𝜆-Calculus") provide representative task-specific realizations for pairwise tasks and multi-hop search. These algorithms show how the abstract fixed-point formulation is turned into a concrete, finite, and auditable execution procedure whose call structure, recursion depth, and composition behavior are fixed after task-type selection and planning, before the first recursive execution step is run.

Algorithm 3 λ​\-RLM\\lambda\\text{-}\\textsf{RLM}: Complete System

1:Prompt P∈Σ∗P\\in\\Sigma^{\*}, model ℳ\\mathcal{M} with window KK, accuracy target α∈(0,1\]\\alpha\\in(0,1\] 

2:Response Y∈Σ∗Y\\in\\Sigma^{\*} 

3:// == Phase 1: REPL Initialization (same as original RLM) == 

4:state←InitRepl​(prompt\=P)\\text{state}\\leftarrow\\textsc{InitRepl}(\\texttt{prompt}=P) ⊳\\triangleright PP lives in environment, not context window 

5:state←RegisterLibrary​(state,ℒ)\\text{state}\\leftarrow\\textsc{RegisterLibrary}(\\text{state},\\,\\mathcal{L}) ⊳\\triangleright load pre-verified combinators into REPL 

6:state←RegisterSubCall​(state,sub\_​ℳ)\\text{state}\\leftarrow\\textsc{RegisterSubCall}(\\text{state},\\,\\texttt{sub\\\_}\\mathcal{M}) ⊳\\triangleright register ℳ\\mathcal{M} as callable, same as RLM 

7:// == Phase 2: Task Detection (1 LLM call) == 

8:meta←ℰ​(state,Peek(P, 0, 500); len(P))\\text{meta}\\leftarrow\\mathcal{E}\\bigl(\\text{state},\\;\\texttt{Peek(P, 0, 500); len(P)}\\bigr) ⊳\\triangleright probe PP via REPL, not neural 

9:τtype←ℳ​(“Select from ​𝒯​: ”∥meta)\\tau\_{\\text{type}}\\leftarrow\\mathcal{M}\\bigl(\\text{\`\`Select from }\\mathcal{T}\\text{: ''}\\|\\text{meta}\\bigr) ⊳\\triangleright menu selection, single call 

10:// == Phase 3: Optimal Planning (0 LLM calls, pure math) == 

11:⊕←Table [1](#S3.T1 "Table 1 ‣ A Compact Combinator Library. ‣ 3.1 From Open-Ended Control to a Restricted Runtime ‣ 3 The 𝜆⁢"-RLM" Framework ‣ The 𝐘-Combinator for LLMs: Solving Long-Context Rot with 𝜆-Calculus")B\[τtype\];π←Table [1](#S3.T1 "Table 1 ‣ A Compact Combinator Library. ‣ 3.1 From Open-Ended Control to a Restricted Runtime ‣ 3 The 𝜆⁢"-RLM" Framework ‣ The 𝐘-Combinator for LLMs: Solving Long-Context Rot with 𝜆-Calculus")B\[τtype\]\\oplus\\leftarrow\\textsc{Table\\,\\ref{tab:combined}}\\textsc{B}\[\\tau\_{\\text{type}}\];\\quad\\pi\\leftarrow\\textsc{Table\\,\\ref{tab:combined}}\\textsc{B}\[\\tau\_{\\text{type}}\] 

12:if |P|≤K|P|\\leq K then 

13: k∗←1;τ∗←|P|k^{\*}\\!\\leftarrow\\!1;\\;\\tau^{\*}\\!\\leftarrow\\!|P| 

14:else 

15: k∗←⌈|P|⋅cin/c⊕⌉k^{\*}\\leftarrow\\big\\lceil\\sqrt{|P|\\cdot c\_{\\text{in}}/c\_{\\oplus}}\\big\\rceil ⊳\\triangleright minimize T​(n)\=k⋅T​(n/k)+𝒞⊕​(k)T(n)=k\\!\\cdot\\!T(n/k)+\\mathcal{C}\_{\\oplus}(k), base T​(τ)\=𝒞​(τ)T(\\tau)=\\mathcal{C}(\\tau) 

16: d←⌈logk∗⁡(|P|/K)⌉d\\leftarrow\\lceil\\log\_{k^{\*}}\\!(|P|/K)\\rceil 

17: while 𝒜​(K)d⋅𝒜⊕d<α\\mathcal{A}(K)^{d}\\!\\cdot\\!\\mathcal{A}\_{\\oplus}^{d}<\\alpha and k∗<|P|/Kk^{\*}<|P|/K do ⊳\\triangleright accuracy constraint 

18: k∗←k∗+1;d←⌈logk∗⁡(|P|/K)⌉k^{\*}\\!\\leftarrow\\!k^{\*}\\!+\\!1;\\;\\;d\\leftarrow\\lceil\\log\_{k^{\*}}\\!(|P|/K)\\rceil 

19: end while 

20: τ∗←min⁡(K,⌊|P|/k∗⌋)\\tau^{\*}\\leftarrow\\min(K,\\,\\lfloor|P|/k^{\*}\\rfloor) 

21:end if 

22:// == Phase 4: Cost Estimation (deterministic, pre-execution) == 

23:C^←(k∗)d⋅𝒞​(τ∗)+d⋅𝒞⊕​(k∗)+𝒞​(500)\\hat{C}\\leftarrow(k^{\*})^{d}\\cdot\\mathcal{C}(\\tau^{\*})+d\\cdot\\mathcal{C}\_{\\oplus}(k^{\*})+\\mathcal{C}(500) ⊳\\triangleright exact pre-execution bound 

24:// == Phase 5: Build and Execute Combinator Chain in REPL == 

25:state←ℰ​(state,Φ \= BuildExecutor(k∗, τ∗, ⊕, π, sub\_ℳ))\\text{state}\\leftarrow\\mathcal{E}\\bigl(\\text{state},\\;\\texttt{$\\Phi$ = BuildExecutor($k^{\*}$, $\\tau^{\*}$, $\\oplus$, $\\pi$, sub\\\_$\\mathcal{M}$)}\\bigr) ⊳\\triangleright register Φ\\Phi in REPL 

26:state←ℰ​(state,result = Φ(P))\\text{state}\\leftarrow\\mathcal{E}\\bigl(\\text{state},\\;\\texttt{result = $\\Phi$(P)}\\bigr) ⊳\\triangleright single execution of Φ\\Phi in REPL 

27:Y←state​\[result\]Y\\leftarrow\\text{state}\[\\texttt{result}\] 

28:return YY 

Algorithm 4 Φ\\Phi: Combinator Executor (registered in REPL, not LLM-generated)

1:Prompt PP (from REPL state), parameters k∗,τ∗,⊕,πk^{\*},\\tau^{\*},\\oplus,\\pi (from planner)

2:Result string

3:function Φ\\Phi(PP)

4: if |P|≤τ∗|P|\\leq\\tau^{\*} then ⊳\\triangleright base case: leaf β\\beta\-reduction 

5: q←Template​\[τtype\].Fmt​(P)q\\leftarrow\\textsc{Template}\[\\tau\_{\\text{type}}\].\\textsc{Fmt}(P) ⊳\\triangleright pre-defined prompt template 

6: return sub\_​ℳ​(q)\\texttt{sub\\\_}\\mathcal{M}(q) ⊳\\triangleright invoke ℳ\\mathcal{M} via REPL’s registered sub-call 

7: else⊳\\triangleright recursive case: all REPL-executed combinators from ℒ\\mathcal{L} 

8: \[P1,…,Pk∗\]←Split​(P,k∗)\[P\_{1},\\ldots,P\_{k^{\*}}\]\\leftarrow\\textsc{Split}(P,\\,k^{\*}) ⊳\\triangleright deterministic, pre-verified 

9: if π\\pi includes Filter then ⊳\\triangleright optional pre-filter for search/multi\_hop 

10: previews←Map(λp.Peek(p,0,⌊τ∗/10⌋),\[P1,…,Pk∗\])\\text{previews}\\leftarrow\\textsc{Map}\\bigl(\\lambda p.\\,\\textsc{Peek}(p,0,\\lfloor\\tau^{\*}/10\\rfloor),\\;\[P\_{1},\\ldots,P\_{k^{\*}}\]\\bigr) 

11: \[P1,…,Pk′\]←Filter​(Relevant,Zip​(\[P1:k∗\],previews))\[P\_{1},\\ldots,P\_{k^{\\prime}}\]\\leftarrow\\textsc{Filter}\\bigl(\\textsc{Relevant},\\;\\textsc{Zip}(\[P\_{1:k^{\*}}\],\\,\\text{previews})\\bigr) 

12: end if 

13: \[R1,…,Rk′\]←Map(λpi.Φ(pi),\[P1,…,Pk′\])\[R\_{1},\\ldots,R\_{k^{\\prime}}\]\\leftarrow\\textsc{Map}\\bigl(\\lambda p\_{i}.\\,\\Phi(p\_{i}),\\;\[P\_{1},\\ldots,P\_{k^{\\prime}}\]\\bigr) ⊳\\triangleright recursive sub-calls 

14: return Reduce​(⊕,\[R1,…,Rk′\])\\textsc{Reduce}\\bigl(\\oplus,\\;\[R\_{1},\\ldots,R\_{k^{\\prime}}\]\\bigr) ⊳\\triangleright deterministic composition 

15: end if 

16:end function 

Algorithm 5  Pairwise Tasks

1:PP, predicate ϕ\\phi, ℳ\\mathcal{M}, k∗,τ∗k^{\*},\\tau^{\*} 

2:Pairs 𝒮⊆ℕ×ℕ\\mathcal{S}\\subseteq\\mathbb{N}\\times\\mathbb{N} 

3:// A: Linear neural - O​(n/K)O(n/K) 

4:\[P1:k∗\]←Split​(P,k∗)\[P\_{1:k^{\*}}\]\\leftarrow\\textsc{Split}(P,k^{\*}) 

5:labels ←Map​(sub\_​ℳcls,P1:k∗)\\leftarrow\\textsc{Map}(\\texttt{sub\\\_}\\mathcal{M}\_{\\text{cls}},\\,P\_{1:k^{\*}}) 

6:L←Parse​(Concat​(labels))L\\leftarrow\\textsc{Parse}(\\textsc{Concat}(\\text{labels})) 

7:// B: Quadratic symbolic - FREE 

8:Q←Filter(ϕ,L.Items())Q\\leftarrow\\textsc{Filter}(\\phi,\\,L.\\textsc{Items}()) 

9:𝒮←{(i,j)∣i,j∈Q,i<j}\\mathcal{S}\\leftarrow\\{(i,j)\\mid i,j\\in Q,\\,i<j\\} 

10:return 𝒮\\mathcal{S} 

Algorithm 6  Multi-Hop Search

1:Corpus \[D1,…,Dm\]\[D\_{1},\\ldots,D\_{m}\], query qq, ℳ\\mathcal{M} 

2:Answer YY 

3:// A: Filter - mostly symbolic 

4:prev ←Map(λD.Peek(D,0,500),D1:m)\\leftarrow\\textsc{Map}(\\lambda D.\\,\\textsc{Peek}(D,0,500),\\,D\_{1:m}) 

5:rel ←Filter​(Match​(q),Zip​(D,prev))\\leftarrow\\textsc{Filter}(\\textsc{Match}(q),\\,\\textsc{Zip}(D,\\text{prev})) 

6:// B: Read - |rel|≪m|\\text{rel}|\\ll m 

7:evi ←Map​(sub\_​ℳext,rel)\\leftarrow\\textsc{Map}(\\texttt{sub\\\_}\\mathcal{M}\_{\\text{ext}},\\,\\text{rel}) 

8:// C: Synthesize - 1 call 

9:Y←sub\_​ℳ​(“answer ”​‖q‖​Concat​(evi))Y\\leftarrow\\texttt{sub\\\_}\\mathcal{M}(\\text{\`\`answer ''}\\|q\\|\\textsc{Concat}(\\text{evi})) 

10:return YY 

Algorithm [1](#alg1 "Algorithm 1 ‣ 3.2 Core Formulation ‣ 3 The 𝜆⁢"-RLM" Framework ‣ The 𝐘-Combinator for LLMs: Solving Long-Context Rot with 𝜆-Calculus") presents the complete λ\\lambda\-RLM system as a finite sequence of phases. As in the original RLM formulation, execution begins by initializing a REPL state in which the prompt PP is stored externally, the trusted combinator library ℒ\\mathcal{L} is registered, and the base model is exposed as a callable leaf oracle through sub\_​ℳ\\texttt{sub\\\_}\\mathcal{M}. The key difference from standard RLM arises immediately after this shared setup: rather than entering an open-ended loop in which the model repeatedly emits arbitrary code, λ\\lambda\-RLM performs a single bounded task-type selection step, followed by deterministic planning and a one-shot execution of a pre-built recursive program. The model is asked to choose a task type from a fixed menu based on a lightweight symbolic probe of the prompt. This keeps neural uncertainty localized to semantic classification, while all subsequent control decisions remain symbolic. Once the task type is selected, the planner determines the execution rule by choosing the task-specific composition operator ⊕\\oplus and execution plan π\\pi, together with the structural parameters (k∗,τ∗,d)(k^{\*},\\tau^{\*},d). Here, k∗k^{\*} controls the branching factor of decomposition, τ∗\\tau^{\*} determines the leaf threshold at which recursion terminates, and dd bounds the resulting recursion depth. The planning phase therefore operationalizes the main design goal of λ\\lambda\-RLM: the shape of execution is fixed before recursive execution begins. Finally, the system builds the combinator chain in the REPL, executes it, and returns the resulting response YY.

Algorithm [4](#alg4 "Algorithm 4 ‣ Appendix D Algorithmic Details ‣ The 𝐘-Combinator for LLMs: Solving Long-Context Rot with 𝜆-Calculus") then defines the executor Φ\\Phi constructed from these planned components. This executor is the concrete realization of the fixed-point program in Eq. ([4](#S3.E4 "In 3.2 Core Formulation ‣ 3 The 𝜆⁢"-RLM" Framework ‣ The 𝐘-Combinator for LLMs: Solving Long-Context Rot with 𝜆-Calculus")). Its behavior is intentionally simple. If the current sub-prompt is already below the threshold τ∗\\tau^{\*}, Φ\\Phi formats it with a task-specific leaf template and calls the base model exactly once. Otherwise, it applies a deterministic recursive pattern: split the input, optionally preview and filter chunks when the task plan requires pruning, recursively process the retained chunks, and combine the resulting partial outputs with Reduce​(⊕,⋅)\\textsc{Reduce}(\\oplus,\\cdot). The recursive structure itself is not model-generated. The only neural operations occur at bounded leaves and, for certain task types, in explicitly specified synthesis steps, while splitting, filtering, traversal, and aggregation are all handled by trusted combinators with fixed semantics.

Algorithms [5](#alg5 "Algorithm 5 ‣ Appendix D Algorithmic Details ‣ The 𝐘-Combinator for LLMs: Solving Long-Context Rot with 𝜆-Calculus") and [6](#alg6 "Algorithm 6 ‣ Appendix D Algorithmic Details ‣ The 𝐘-Combinator for LLMs: Solving Long-Context Rot with 𝜆-Calculus") illustrate how this general executor specializes to structured task families. The pairwise algorithm shows that the expensive neural portion can remain linear in the number of chunks: the model is used only to label or extract candidate items, after which the quadratic pairing step is computed symbolically at essentially zero additional neural cost. The multi-hop search algorithm follows the same principle in a different form. It first uses symbolic preview-based filtering to narrow a large corpus to a small relevant subset, then applies neural reading only to that subset, and finally performs a single synthesis step over the extracted evidence. These examples are not separate learning algorithms, but concrete instantiations of the same λ\\lambda\-RLM design principle: use the model only where semantic inference is needed, and realize the surrounding control flow through typed symbolic composition.

BETA

[ ](javascript:toggleReadingMode%28%29; "Disable reading mode, show header and footer") 