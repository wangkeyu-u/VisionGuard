CREATE VIEW ranked_experiments AS
SELECT
    experiment,
    label,
    model,
    imgsz,
    best_epoch,
    validation_map50_95,
    validation_map50,
    test_map50_95,
    mean_latency_ms,
    p95_latency_ms,
    model_size_mib
FROM experiment_metrics
ORDER BY validation_map50_95 DESC;

CREATE VIEW per_class_long AS
SELECT
    class_order,
    class_name,
    'Baseline' AS experiment,
    baseline_ap50_95 AS ap50_95,
    0.0 AS delta
FROM per_class_metrics
UNION ALL
SELECT
    class_order,
    class_name,
    'Selected' AS experiment,
    selected_ap50_95 AS ap50_95,
    delta_ap50_95 AS delta
FROM per_class_metrics
ORDER BY class_order, experiment;

CREATE VIEW comparison_headline AS
SELECT
    selected.validation_map50_95 AS selected_validation_map50_95,
    selected.validation_map50_95 - baseline.validation_map50_95 AS validation_delta,
    selected.test_map50_95 AS selected_test_map50_95,
    selected.test_map50_95 - baseline.test_map50_95 AS test_delta,
    MAX(
        CASE WHEN classes.class_name = 'no_helmet' THEN classes.selected_ap50_95 END
    ) AS no_helmet_ap50_95,
    MAX(
        CASE WHEN classes.class_name = 'no_helmet' THEN classes.delta_ap50_95 END
    ) AS no_helmet_delta,
    MAX(
        CASE WHEN classes.class_name = 'no_vest' THEN classes.selected_ap50_95 END
    ) AS no_vest_ap50_95,
    MAX(
        CASE WHEN classes.class_name = 'no_vest' THEN classes.delta_ap50_95 END
    ) AS no_vest_delta,
    selected.mean_latency_ms AS mean_latency_ms
FROM comparison_context AS context
JOIN experiment_metrics AS baseline ON baseline.experiment = context.baseline_experiment
JOIN experiment_metrics AS selected ON selected.experiment = context.selected_experiment
CROSS JOIN per_class_metrics AS classes
GROUP BY
    selected.validation_map50_95,
    baseline.validation_map50_95,
    selected.test_map50_95,
    baseline.test_map50_95,
    selected.mean_latency_ms;
