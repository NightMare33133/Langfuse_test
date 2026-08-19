"""Evaluation package for batch querying, LLM judging, experiments, and reporting."""

from .batch_query import (
    run_batch_query,
)
from .judge import (
    judge_sample,
    judge_batch,
    calculate_rubric_scores,
)
from .experiment import (
    Experiment,
    ExperimentRun,
    load_experiments,
    save_experiment,
)
from .optimization_analysis import (
    analyze_bad_cases,
    generate_optimization_suggestions,
)
from .retrieval_diff import (
    compare_retrieval_runs,
)
from .report_export import (
    export_experiment_report_excel,
    export_experiment_report_html,
)
